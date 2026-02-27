"""
forensic_features.py
====================
Research-grade forensic feature extractor for JPEG images processed through
social-media compression chains.

Feature vector layout (fixed 258-dimensional NumPy array):
  [0:21]    DCT AC histogram bins
  [21:42]   DCT coefficient energy map
  [42:123]  Intra-block Markov transition features (T=4, 81 dims)
  [123:187] Luminance Q-table (64 dims)
  [187:191] Luma Q-table stats: [mean, std, min, max]
  [191:231] Chrominance Q-table (40 dims)
  [231:235] Chroma Q-table stats: [mean, std, min, max]
  [235:241] Metadata flags (6 dims)
  [241:247] Structural stats (6 dims)
  [247:252] Double-compression ghost peaks (AC1-AC5 empty bins ratio, 5 dims)
  [252:258] Q-Table Backtracking (L1 distance to 6 standard tables, 6 dims)

Total: 258 dimensions
"""

import numpy as np
import math
import io
import os
import struct
from PIL import Image, ExifTags

# ─────────────────────────────────────────────
#  Zigzag scan order for 8x8 DCT block
# ─────────────────────────────────────────────
ZIGZAG_ORDER = [
    (0,0),(0,1),(1,0),(2,0),(1,1),(0,2),(0,3),(1,2),
    (2,1),(3,0),(4,0),(3,1),(2,2),(1,3),(0,4),(0,5),
    (1,4),(2,3),(3,2),(4,1),(5,0),(6,0),(5,1),(4,2),
    (3,3),(2,4),(1,5),(0,6),(0,7),(1,6),(2,5),(3,4),
    (4,3),(5,2),(6,1),(7,0),(7,1),(6,2),(5,3),(4,4),
    (3,5),(2,6),(1,7),(2,7),(3,6),(4,5),(5,4),(6,3),
    (7,2),(7,3),(6,4),(5,5),(4,6),(3,7),(4,7),(5,6),
    (6,5),(7,4),(7,5),(6,6),(5,7),(6,7),(7,6),(7,7),
]
AC_ZIGZAG = ZIGZAG_ORDER[1:22]

FEATURE_DIM = 258

Q_TABLE_LIBRARY = {
    '2024 Facebook': np.array([4, 3, 3, 4, 3, 3, 4, 4, 3, 4, 5, 4, 4, 5, 6, 10, 7, 6, 6, 6, 6, 13, 9, 10, 8, 10, 15, 13, 16, 16, 15, 13, 15, 14, 17, 19, 24, 20, 17, 18, 23, 18, 14, 15, 21, 28, 21, 23, 25, 25, 27, 27, 27, 16, 20, 29, 31, 29, 26, 31, 24, 26, 27, 26], dtype=np.float32),
    '2024 Flickr': np.array([4, 3, 3, 4, 3, 3, 4, 4, 3, 4, 4, 4, 4, 5, 6, 10, 7, 6, 6, 6, 6, 13, 9, 10, 7, 10, 15, 13, 16, 16, 15, 13, 14, 14, 17, 19, 24, 20, 17, 17, 23, 18, 14, 14, 21, 29, 21, 23, 25, 26, 27, 27, 27, 16, 20, 29, 32, 29, 26, 31, 24, 26, 27, 26], dtype=np.float32),
    '2024 Twitter': np.array([5, 3, 4, 4, 4, 3, 5, 4, 4, 4, 5, 5, 5, 6, 7, 12, 8, 7, 7, 7, 7, 15, 11, 11, 9, 12, 17, 15, 18, 18, 17, 15, 17, 17, 19, 22, 28, 23, 19, 20, 26, 21, 17, 17, 24, 33, 24, 26, 29, 29, 31, 31, 31, 19, 23, 34, 36, 34, 30, 36, 28, 30, 31, 30], dtype=np.float32),
    '2026 Telegram': np.array([4, 3, 3, 4, 3, 3, 4, 4, 3, 4, 5, 4, 4, 5, 6, 10, 7, 6, 6, 6, 6, 13, 9, 10, 8, 10, 15, 13, 16, 16, 15, 13, 15, 14, 17, 19, 24, 20, 17, 18, 23, 18, 14, 15, 21, 28, 21, 23, 25, 25, 27, 27, 27, 16, 20, 29, 31, 29, 26, 31, 24, 26, 27, 26], dtype=np.float32),
    '2026 Slack': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 2, 2, 2, 2, 2, 4, 3, 3, 2, 3, 5, 4, 5, 5, 5, 4, 4, 4, 5, 6, 7, 6, 5, 5, 7, 6, 4, 4, 6, 9, 6, 7, 8, 8, 8, 8, 8, 5, 6, 9, 10, 9, 8, 10, 7, 8, 8, 8], dtype=np.float32),
    '2026 Discord': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 2, 2, 2, 2, 2, 4, 3, 3, 2, 4, 4, 5, 6, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 5, 5, 7, 7, 7, 7, 7, 8, 8, 8, 8, 6, 7, 9, 9, 9, 9, 9, 9, 9, 9, 9], dtype=np.float32),
}

def _dct_matrix_8():
    mat = np.zeros((8, 8))
    for u in range(8):
        for v in range(8):
            alpha = 1.0 / math.sqrt(2) if u == 0 else 1.0
            mat[u, v] = alpha * math.cos((2 * v + 1) * u * math.pi / 16.0)
    return mat * math.sqrt(2.0 / 8)

_DCT_MATRIX = _dct_matrix_8()

def _parse_jpeg_header(path: str) -> dict:
    info = {"q_tables": {}, "is_progressive": False, "width": 0, "height": 0}
    try:
        with open(path, "rb") as f:
            data = f.read()
        i = 0
        if data[0:2] != b'\xff\xd8': return info
        i = 2
        while i < len(data) - 1:
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i:i+2]
            i += 2
            if marker == b'\xff\xd9': break
            if marker in (b'\xff\xd8', b'\xff\x01'): continue
            if i + 2 > len(data): break
            length = struct.unpack(">H", data[i:i+2])[0]
            segment = data[i+2:i+length]
            i += length

            if marker == b'\xff\xdb':
                j = 0
                while j < len(segment):
                    qt_info = segment[j]
                    precision = (qt_info >> 4) & 0xF
                    table_id = qt_info & 0xF
                    j += 1
                    table_bytes = 64 if precision == 0 else 128
                    if precision == 0:
                        qt = np.array(list(segment[j:j+table_bytes]), dtype=np.float32)
                    else:
                        qt = np.array([struct.unpack(">H", segment[j+k*2:j+k*2+2])[0] for k in range(64)], dtype=np.float32)
                    j += table_bytes
                    info["q_tables"][table_id] = qt
            elif marker == b'\xff\xc0':
                if len(segment) >= 5:
                    info["height"] = struct.unpack(">H", segment[1:3])[0]
                    info["width"]  = struct.unpack(">H", segment[3:5])[0]
            elif marker == b'\xff\xc2':
                info["is_progressive"] = True
                if len(segment) >= 5:
                    info["height"] = struct.unpack(">H", segment[1:3])[0]
                    info["width"]  = struct.unpack(">H", segment[3:5])[0]
    except Exception:
        pass
    return info

class ForensicFeatureExtractor:
    def __init__(self, num_blocks_sample: int = 2000):
        self.num_blocks_sample = num_blocks_sample

    def extract(self, image_path: str) -> np.ndarray:
        vec = np.zeros(FEATURE_DIM, dtype=np.float32)
        try:
            header = _parse_jpeg_header(image_path)
            file_size = os.path.getsize(image_path)

            img = Image.open(image_path)
            if img.mode not in ("RGB", "L", "YCbCr"): img = img.convert("RGB")
            width, height = img.size

            img_ycbcr = img.convert("YCbCr")
            y_full = np.array(img_ycbcr, dtype=np.float32)[:, :, 0]

            h, w = y_full.shape
            y = y_full[:(h//8)*8, :(w//8)*8]
            blocks = y.reshape(h//8, 8, w//8, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
            n_blocks = len(blocks)

            if n_blocks > self.num_blocks_sample:
                rng = np.random.default_rng(seed=42)
                blocks = blocks[rng.choice(n_blocks, self.num_blocks_sample, replace=False)]

            blocks_c = blocks - 128.0
            D = _DCT_MATRIX
            dct_blocks = D @ blocks_c @ D.T

            # 1. DCT AC Histogram & 2. Energy
            ac_means = np.zeros(21, dtype=np.float32)
            for k, (u, v) in enumerate(AC_ZIGZAG):
                ac_means[k] = float(np.mean(np.abs(dct_blocks[:, u, v])))
            g_max = np.max(ac_means) if np.max(ac_means) > 0 else 1.0
            vec[0:21] = ac_means / g_max

            dc_energy = max(float(np.mean(np.abs(dct_blocks[:, 0, 0]))), 1.0)
            ac_energy = np.zeros(21, dtype=np.float32)
            for k, (u, v) in enumerate(AC_ZIGZAG):
                ac_energy[k] = float(np.mean(np.abs(dct_blocks[:, u, v]))) / dc_energy
            vec[21:42] = np.clip(ac_energy, 0, 1)

            # 3. Intra-block Markov transitions (T=4) -> 81 dims
            # Quantize AC coefficients to integers in range [-4, 4].
            # This captures spatial adjacency probability (Markov matrix) for block shape boundaries,
            # which is highly resistant to subsequent JPEG quantizations because it measures structure rather than raw values.
            intra_markov = np.zeros((9, 9), dtype=np.float32)
            # Take first 21 ACs for intra-block transitions
            for k in range(20):
                u1, v1 = AC_ZIGZAG[k]
                u2, v2 = AC_ZIGZAG[k+1]
                c1 = np.clip(np.round(dct_blocks[:, u1, v1]), -4, 4).astype(int) + 4
                c2 = np.clip(np.round(dct_blocks[:, u2, v2]), -4, 4).astype(int) + 4
                for v_from, v_to in zip(c1, c2):
                    intra_markov[v_from, v_to] += 1
            
            row_sums = intra_markov.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            intra_markov /= row_sums
            vec[42:123] = intra_markov.flatten()

            # 4. Luma Q-table & Stats
            luma_qt = header["q_tables"].get(0, getattr(img, "quantization", {}).get(0, None))
            if luma_qt is not None and len(luma_qt) >= 64:
                qt_vals = np.array(luma_qt[:64], dtype=np.float32)
                vec[123:187] = qt_vals / 255.0
                vec[187] = float(np.mean(qt_vals)) / 255.0
                vec[188] = float(np.std(qt_vals)) / 255.0
                vec[189] = float(np.min(qt_vals)) / 255.0
                vec[190] = float(np.max(qt_vals)) / 255.0
            else:
                qt_vals = np.zeros(64, dtype=np.float32) # For L1 dist math

            # 5. Chroma Q-table & Stats
            chroma_qt = header["q_tables"].get(1, getattr(img, "quantization", {}).get(1, None))
            if chroma_qt is not None and len(chroma_qt) >= 40:
                cq = np.array(chroma_qt[:40], dtype=np.float32)
                vec[191:231] = cq / 255.0
                vec[231] = float(np.mean(chroma_qt)) / 255.0
                vec[232] = float(np.std(chroma_qt)) / 255.0
                vec[233] = float(np.min(chroma_qt)) / 255.0
                vec[234] = float(np.max(chroma_qt)) / 255.0

            # 6. Meta
            exif_data = img._getexif() if hasattr(img, '_getexif') else None
            if exif_data:
                exif = {ExifTags.TAGS.get(k, k): v for k, v in exif_data.items()}
                vec[235] = 1.0
                vec[236] = 1.0 if "DateTimeOriginal" in exif else 0.0
                vec[237] = float(abs(hash(str(exif.get("Software", "")).strip())) % 1_000_000) / 1e6
            vec[238] = 1.0 if len(header["q_tables"]) > 0 else 0.0
            vec[239] = min(float(np.sum(header["q_tables"][0])) / 6400.0, 1.0) if 0 in header["q_tables"] else 0.0
            vec[240] = 1.0 if header["is_progressive"] else 0.0

            # 7. Structural
            hw, ww = header.get("height", height) or height, header.get("width", width) or width
            vec[241] = min(ww / 10000.0, 1.0)
            vec[242] = min(hw / 10000.0, 1.0)
            vec[243] = min(file_size / max(ww * hw, 1) / 10.0, 1.0)
            vec[244] = float(np.clip(np.mean(y_full) / 255.0, 0, 1))
            vec[245] = float(np.clip(np.std(y_full)  / 255.0, 0, 1))
            
            hb = y_full[7::8, :]
            hn = y_full[8::8, :]
            min_r = min(len(hb), len(hn))
            bbd = float(np.mean(np.abs(hb[:min_r].astype(float) - hn[:min_r].astype(float)))) / 255.0 if min_r > 0 else 0.0
            vec[246] = float(np.clip(bbd, 0, 1))

            # 8. Ghost Peaks (Empty Bins AC1-AC5)
            # Detects "Double Compression Artifacts". When a signal is quantized by step Q1 
            # and then re-quantized by Q2 (where Q1 < Q2), periodic gaps ("ghost peaks") 
            # appear in the resulting DCT histogram. We measure the % of empty bins.
            ghost_peaks = np.zeros(5, dtype=np.float32)
            for k in range(5):
                u, v = AC_ZIGZAG[k]
                c = np.round(dct_blocks[:, u, v])
                c_min, c_max = int(np.min(c)), int(np.max(c))
                span = c_max - c_min
                if span > 0:
                    hist, _ = np.histogram(c, bins=np.arange(c_min, c_max + 2))
                    ghost_peaks[k] = float(np.sum(hist == 0)) / float(span)
            vec[247:252] = ghost_peaks

            # 9. Q-Table Backtracking (L1 Distances)
            # Distance calculated against denormalized Qt_vals (0-255)
            # Then we divide by (64*255) to normalize L1 dist to [0,1]. Small = high match.
            q_dist = np.zeros(6, dtype=np.float32)
            lib_keys = ['2024 Facebook', '2024 Flickr', '2024 Twitter', '2026 Telegram', '2026 Slack', '2026 Discord']
            for i, name in enumerate(lib_keys):
                t_ref = Q_TABLE_LIBRARY[name]
                l1 = float(np.sum(np.abs(qt_vals - t_ref)))
                q_dist[i] = l1 / (64.0 * 255.0)
            vec[252:258] = q_dist

        except Exception as e:
            # print(e)
            pass

        return vec

    @property
    def feature_dim(self) -> int:
        return FEATURE_DIM

    def feature_names(self) -> list:
        names = []
        for i in range(21): names.append(f"dct_ac_hist_{i+1}")
        for i in range(21): names.append(f"dct_ac_energy_{i+1}")
        for i in range(9):
            for j in range(9): names.append(f"intra_markov_{i-4}_to_{j-4}")
        for i in range(64): names.append(f"luma_qt_{i}")
        names += ["luma_qt_mean", "luma_qt_std", "luma_qt_min", "luma_qt_max"]
        for i in range(40): names.append(f"chroma_qt_{i}")
        names += ["chroma_qt_mean", "chroma_qt_std", "chroma_qt_min", "chroma_qt_max"]
        names += [
            "meta_has_exif", "meta_has_datetime", "meta_software_hash",
            "meta_has_qt", "meta_qt0_sum_norm", "meta_is_progressive",
            "struct_width", "struct_height", "struct_compression",
            "struct_y_mean", "struct_y_std", "struct_bbd",
        ]
        for i in range(5): names.append(f"ghost_peak_ac{i+1}")
        names += [
            "qdist_24FB", "qdist_24FL", "qdist_24TW",
            "qdist_26TG", "qdist_26SL", "qdist_26DC"
        ]
        assert len(names) == FEATURE_DIM, f"Count {len(names)} != {FEATURE_DIM}"
        return names
