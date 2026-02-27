#!/usr/bin/env python3
"""
chained_uploader_fixed_keys.py

Chained upload/download orchestrator for Telegram, Slack, Reddit.
All service tokens and access keys are defined as global fixed variables
for local student testing. Do not commit this file to any public repo.
"""

import os
import io
import time
import json
import csv
import random
import hashlib
import argparse
import tempfile
from pathlib import Path
from itertools import permutations

import requests
from PIL import Image, ImageFile

# Telegram
from telegram import Bot
from telegram.request import HTTPXRequest

# Slack
from slack_sdk import WebClient

# Discord
import discord
import asyncio

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------- Fixed global credentials (hardcoded for local testing) ----------
# Replace the placeholder strings below with your actual tokens/IDs.
# WARNING: This file contains secrets in plaintext. Do not commit to VCS.

# Telegram
TELEGRAM_BOT_TOKEN = "replace_with_your_telegram_bot_token"
TELEGRAM_CHAT_ID = "replace_with_your_telegram_chat_id"

# Slack
SLACK_BOT_TOKEN = "replace_with_your_slack_bot_token"
SLACK_CHANNEL_ID = "replace_with_your_slack_channel_id"

# Discord
DISCORD_BOT_TOKEN = "replace_with_your_discord_bot_token"
DISCORD_CHANNEL_ID = "replace_with_your_discord_channel_id"

# -------------------------------------------------------------------------

# ---------- Helpers ----------
def sha256_bytes(b: bytes) -> str:
    import hashlib
    return hashlib.sha256(b).hexdigest()

def save_bytes(path: Path, b: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b)

def read_bytes(path: Path) -> bytes:
    return path.read_bytes()

def now_ts():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

# ---------- Platform adapters ----------
class TelegramAdapter:
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id

    def upload_and_download(self, image_path: Path, image_bytes: bytes = None):
        return asyncio.run(self._process(image_path, image_bytes))

    async def _process(self, image_path, image_bytes):
        # send_photo limit: 10MB
        MAX_TELEGRAM_SIZE = 10 * 1024 * 1024
        
        data_to_send = image_bytes
        if data_to_send is None:
            data_to_send = image_path.read_bytes()
            
        if len(data_to_send) > MAX_TELEGRAM_SIZE:
            print(f"  [Telegram] Image size {len(data_to_send)} > 10MB. Compressing...")
            # Compress logic
            try:
                img = Image.open(io.BytesIO(data_to_send))
                quality = 98
                while quality > 10:
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=quality)
                    size = buf.tell()
                    if size < MAX_TELEGRAM_SIZE:
                        data_to_send = buf.getvalue()
                        print(f"  [Telegram] Compressed to {size} bytes (quality={quality})")
                        break
                    quality -= 2
            except Exception as e:
                 print(f"  [Telegram] Compression failed: {e}")

        fobj = io.BytesIO(data_to_send)
        
        try:
            # High timeout request
            req = HTTPXRequest(connection_pool_size=8, read_timeout=120.0, write_timeout=120.0, connect_timeout=60.0)
            bot = Bot(token=self.token, request=req)
            async with bot:
                # Retry loop for sending
                msg = None
                for attempt in range(3):
                    try:
                        msg = await bot.send_photo(chat_id=self.chat_id, photo=fobj, read_timeout=120, write_timeout=120, connect_timeout=60)
                        break
                    except Exception as e:
                        print(f"  [Telegram] Send retry {attempt+1}/3 failed: {e}")
                        if attempt == 2: raise
                        await asyncio.sleep(5)
                        fobj.seek(0)
                
                # pick largest size
                photo_sizes = msg.photo
                file_id = photo_sizes[-1].file_id
                
                new_file = await bot.get_file(file_id, read_timeout=120, connect_timeout=60)
                
                # download_to_memory
                out_buffer = io.BytesIO()
                await new_file.download_to_memory(out_buffer)
                out_buffer.seek(0)
                served = out_buffer.read()
                
                meta = {"platform": "telegram", "file_id": file_id}
                return bytes(served), meta
        finally:
            if hasattr(fobj, "close"):
                fobj.close()

class SlackAdapter:
    def __init__(self, token, channel_id):
        self.client = WebClient(token=token, timeout=180)
        self.channel = channel_id
        self.token = token

    def upload_and_download(self, image_path: Path, image_bytes: bytes = None):
        # Slack-specific pre-compression: cap at 4MB to avoid IncompleteRead on large images
        SLACK_MAX_BYTES = 4 * 1024 * 1024  # 4MB
        if image_bytes is not None and len(image_bytes) > SLACK_MAX_BYTES:
            image_bytes = compress_if_needed(image_bytes, max_size=SLACK_MAX_BYTES)
        elif image_bytes is None:
            raw = image_path.read_bytes()
            if len(raw) > SLACK_MAX_BYTES:
                image_bytes = compress_if_needed(raw, max_size=SLACK_MAX_BYTES)

        # Slack files_upload is deprecated, use v2
        if image_bytes is None:
            resp = self.client.files_upload_v2(channel=self.channel, file=str(image_path), filename=image_path.name)
        else:
            resp = self.client.files_upload_v2(channel=self.channel, file=io.BytesIO(image_bytes), filename=image_path.name)
        file_id = resp["file"]["id"]
        info = self.client.files_info(file=file_id)
        url = info["file"].get("url_private_download") or info["file"].get("url_private")
        headers = {"Authorization": f"Bearer {self.token}"}
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        meta = {"platform": "slack", "file_id": file_id, "url": url}
        return r.content, meta

import threading
import queue

class DiscordAdapter:
    def __init__(self, token, channel_id):
        self.token = token
        try:
            self.channel_id = int(channel_id)
        except ValueError:
            self.channel_id = 0
        
        self.loop = asyncio.new_event_loop()
        self.ready_event = threading.Event()
        self.thread = threading.Thread(target=self._start_loop, daemon=True)
        self.thread.start()
        
        # Wait for client to be ready (optional, but good for safety)
        # We can't easily wait for "on_ready" here without blocking, 
        # but the client creation is fast. The connection happens in background.

    def _start_loop(self):
        asyncio.set_event_loop(self.loop)
        intents = discord.Intents.default()
        self.client = discord.Client(intents=intents)

        @self.client.event
        async def on_ready():
            self.ready_event.set()
            print(f"Discord Client connected as {self.client.user}")

        self.loop.run_until_complete(self.client.start(self.token))

    def upload_and_download(self, image_path: Path, image_bytes: bytes = None):
        # Schedule the async task on the background loop
        if not self.ready_event.is_set():
             # Basic wait if called too early
             # print("Waiting for Discord connection...")
             self.ready_event.wait(timeout=10)

        future = asyncio.run_coroutine_threadsafe(
            self._process(image_path, image_bytes), self.loop
        )
        return future.result()

    async def _process(self, image_path, image_bytes):
        result = {}
        try:
            channel = self.client.get_channel(self.channel_id)
            if not channel:
                # Try fetching if not in cache
                try:
                    channel = await self.client.fetch_channel(self.channel_id)
                except Exception as e:
                    raise RuntimeError(f"Failed to fetch channel {self.channel_id}: {e}")

            # Prepare file
            if image_bytes is None:
                f = discord.File(str(image_path), filename=image_path.name)
            else:
                f = discord.File(io.BytesIO(image_bytes), filename=image_path.name)
            
            # Send
            msg = await channel.send(file=f)
            
            # Wait for attachment to be fully processed
            # Retry loop for attachment availability
            max_retries = 5
            att = None
            for _ in range(max_retries):
                # Refresh message
                msg = await channel.fetch_message(msg.id)
                if msg.attachments:
                     att = msg.attachments[0]
                     # Check if height/width are present effectively means processed
                     if att.width or att.height or att.size:
                         break
                await asyncio.sleep(2)
            
            if not att:
                raise RuntimeError("No attachments found or processed in sent message.")
            
            # Use 'url' (CDN) instead of 'proxy_url'
            url = att.url 
            msg_id = msg.id
            
            # Better: use loop.run_in_executor for the download to avoid blocking the bot heartbeat
            loop = asyncio.get_running_loop()
            r = await loop.run_in_executor(None, lambda: requests.get(url, timeout=120))
            r.raise_for_status()
            
            meta = {"platform": "discord", "msg_id": msg_id, "url": url}
            return r.content, meta

        except Exception as e:
            raise RuntimeError(f"Discord processing failed: {e}")

# ---------- Orchestrator ----------
DEFAULT_PLATFORMS = ["telegram", "slack", "discord"]

def build_sequences(platforms, k, mode, chains_per_image):
    seqs = []
    if mode == "permutations":
        for perm in permutations(platforms, k):
            seqs.append(list(perm))
    else:  # random
        for _ in range(chains_per_image):
            seq = random.sample(platforms, k)
            seqs.append(seq)
    return seqs

# ---------- Helpers ----------
def compress_if_needed(image_bytes: bytes, max_size=7.9 * 1024 * 1024):
    if len(image_bytes) <= max_size:
        return image_bytes
    
    print(f"  [Safeguard] Image size {len(image_bytes)} > {max_size}. Compressing...")
    try:
        img = Image.open(io.BytesIO(image_bytes))
        quality = 98
        while quality > 10:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            size = buf.tell()
            if size <= max_size:
                print(f"  [Safeguard] Compressed to {size} bytes (quality={quality})")
                return buf.getvalue()
            quality -= 2
        return image_bytes # Failed to compress enough, return original (will likely fail)
    except Exception as e:
        print(f"  [Safeguard] Compression failed: {e}")
        return image_bytes

def run_chain_for_image(orig_path: Path, seq: list, adapters: dict, outdir: Path, delay=3):
    manifest_entries = []
    current_bytes = orig_path.read_bytes()
    
    # Global Forensic Safeguard: 7.9MB limit
    current_bytes = compress_if_needed(current_bytes)
    
    current_name = orig_path.name
    chain_id = sha256_bytes(current_bytes)[:8] + "_" + str(int(time.time()))
    step = 0
    
    # Strict Chain Validation
    chain_failed = False
    written_files = []
    
    for platform in seq:
        if chain_failed:
            break
            
        step += 1
        adapter = adapters.get(platform)
        if adapter is None:
            raise RuntimeError(f"No adapter for {platform}")
            
        try:
            served_bytes, meta = adapter.upload_and_download(orig_path, image_bytes=current_bytes)
        except Exception as e:
            print(f"  [ERROR] Platform {platform} failed: {e}")
            chain_failed = True
            break # Stop chain immediately
            
        ts = now_ts()
        entry = {
            "chain_id": chain_id,
            "orig_image": orig_path.name,
            "step": step,
            "platform": platform,
            "timestamp": ts,
            "meta": meta
        }
        
        if served_bytes:
            fname = f"{orig_path.stem}.chain_{chain_id}.step{step}.{platform}{orig_path.suffix}"
            outpath = outdir / fname
            save_bytes(outpath, served_bytes)
            written_files.append(outpath)
            
            entry.update({
                "served_filename": str(outpath.name),
                "served_sha256": sha256_bytes(served_bytes),
                "served_size": len(served_bytes)
            })
            # prepare for next step
            current_bytes = served_bytes
        else:
            # Should not happen if served_bytes is None implies failure caught above, 
            # unless adapter returns None, None for some reason?
            chain_failed = True
            break

        manifest_entries.append(entry)
        
        # Stability tweak: Random sleep
        sleep_time = random.uniform(5, 12)
        print(f"  Sleeping {sleep_time:.2f}s...")
        time.sleep(sleep_time)
        
    if chain_failed:
        print(f"  [Safeguard] Chain {seq} failed. Deleting {len(written_files)} partial files.")
        for f in written_files:
            try:
                if f.exists(): f.unlink()
            except Exception as e:
                print(f"  [Error] Failed to delete {f.name}: {e}")
        return [] # Return empty list to discard partial data
        
    return manifest_entries

# ---------- Main ----------
def main(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    images = [Path(p) for p in args.images]
    platforms = args.platforms.split(",") if args.platforms else DEFAULT_PLATFORMS
    # init adapters using fixed global credentials
    adapters = {}
    if "telegram" in platforms:
        adapters["telegram"] = TelegramAdapter(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    if "slack" in platforms:
        adapters["slack"] = SlackAdapter(SLACK_BOT_TOKEN, SLACK_CHANNEL_ID)
    if "discord" in platforms:
        adapters["discord"] = DiscordAdapter(DISCORD_BOT_TOKEN, DISCORD_CHANNEL_ID)

    all_manifest = []
    # --- Paired Strategy Selection Logic ---
    images = []
    
    # 1. RAISE (first 50)
    raise_dir = Path("RAISE_pristine")
    if raise_dir.exists():
        raise_imgs = sorted([p for p in raise_dir.glob("*.jpg")])[:50]
        print(f"Found {len(raise_imgs)} RAISE images.")
        images.extend(raise_imgs)
    else:
        print("Warning: RAISE_pristine directory not found.")

    # 2. VISION (10 per folder)
    vision_dir = Path("VISION_source")
    if vision_dir.exists():
        vision_count = 0
        # Iterate over camera folders
        for cam_folder in sorted(vision_dir.iterdir()):
            if cam_folder.is_dir():
                cam_imgs = sorted([p for p in cam_folder.glob("*.jpg")])[:10]
                images.extend(cam_imgs)
                vision_count += len(cam_imgs)
        print(f"Found {vision_count} VISION images.")
    else:
        print("Warning: VISION_source directory not found.")
        
    print(f"Total images found: {len(images)}")
    
    expected_files = 0
    # Calculate expected files per image to determine completion
    test_seqs = build_sequences(platforms, args.chain_length, args.mode, args.chains_per_image)
    files_per_chain = args.chain_length
    expected_files = len(test_seqs) * files_per_chain
    print(f"Resume logic: Expecting {expected_files} files per completed image.")

    if args.resume:
        print("Resume mode enabled. Scanning for processed images...")
        stem_counts = {}
        for p in outdir.glob("*"):
             if ".chain_" in p.name and p.suffix.lower() == ".jpg":
                stem = p.name.split(".chain_")[0]
                stem_counts[stem] = stem_counts.get(stem, 0) + 1
        
        # Only consider processed if we have at least the expected number of files
        # (Note: if multiple runs happened, we might have > expected, which is fine)
        processed_stems = {s for s, c in stem_counts.items() if c >= expected_files}
        
        original_count = len(images)
        images = [img for img in images if img.stem not in processed_stems]
        print(f"Skipping {original_count - len(images)} completed images. Remaining to process: {len(images)}")
    else:
        print(f"Total images to process: {len(images)}")

    all_manifest = []
    
    # Batching logic
    BATCH_SIZE = 5
    PAUSE_TIME = 60
    
    for i in range(0, len(images), BATCH_SIZE):
        batch = images[i:i + BATCH_SIZE]
        print(f"\n--- Processing Batch {i//BATCH_SIZE + 1} ({len(batch)} images) ---")
        
        for img in batch:
            seqs = build_sequences(platforms, args.chain_length, args.mode, args.chains_per_image)
            for j, seq in enumerate(seqs, start=1):
                print(f"Image {img.name} chain {j}/{len(seqs)}: {seq}")
                try:
                    entries = run_chain_for_image(img, seq, adapters, outdir) # delay handled inside
                    # annotate sequence and index
                    for e in entries:
                        e["sequence_index"] = j
                        e["sequence"] = seq
                    all_manifest.extend(entries)
                except Exception as e:
                    print(f"Failed chain for {img.name}: {e}")

        if i + BATCH_SIZE < len(images):
            print(f"Batch complete. Pausing for {PAUSE_TIME} seconds to clear rate limits...")
            time.sleep(PAUSE_TIME)

    # save manifest JSON and CSV
    manifest_json = outdir / "manifest.json"
    manifest_csv = outdir / "manifest.csv"
    manifest_json.write_text(json.dumps(all_manifest, indent=2))
    # CSV flatten
    keys = ["chain_id", "orig_image", "sequence_index", "sequence", "step", "platform",
            "timestamp", "served_filename", "served_sha256", "served_size", "meta"]
    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in all_manifest:
            flat = {k: row.get(k) for k in keys}
            # ensure sequence is string
            flat["sequence"] = ",".join(row.get("sequence", [])) if row.get("sequence") else ""
            flat["meta"] = json.dumps(row.get("meta", {}))
            writer.writerow(flat)
    print("Done. Results in", outdir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chained uploader for forensic sharing simulation")
    parser.add_argument("--images", nargs="+", required=True, help="Input image paths")
    parser.add_argument("--chain-length", type=int, default=2, help="Number of platforms per chain (k)")
    parser.add_argument("--chains-per-image", type=int, default=5, help="Number of random chains per image (used in random mode)")
    parser.add_argument("--mode", choices=["random", "permutations"], default="random", help="Sequence generation mode")
    parser.add_argument("--platforms", default="telegram,slack,discord", help="Comma-separated platforms to use")
    parser.add_argument("--outdir", default="results", help="Output directory")
    parser.add_argument("--delay", type=float, default=3.0, help="Delay (s) between steps to be polite")
    parser.add_argument("--resume", action="store_true", help="Skip images that have already been processed (found in outdir)")
    args = parser.parse_args()
    main(args)
