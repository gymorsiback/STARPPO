import math
import os
import json
import time
import uuid
from datetime import datetime
from typing import List, Tuple, Dict, Any

import numpy as np


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def latest_symlink(target_path: str, latest_path: str):
    try:
        if os.path.islink(latest_path) or os.path.exists(latest_path):
            try:
                os.remove(latest_path)
            except IsADirectoryError:
                import shutil
                shutil.rmtree(latest_path)
        os.symlink(os.path.basename(target_path), latest_path)
    except Exception:
        import shutil
        shutil.copyfile(target_path, latest_path)


def moving_average(x: List[float], w: int) -> np.ndarray:
    if w <= 1 or len(x) == 0:
        return np.array(x, dtype=float)
    cumsum = np.cumsum(np.insert(np.array(x, dtype=float), 0, 0))
    ma = (cumsum[w:] - cumsum[:-w]) / float(w)
    pad_left = [ma[0]] * (w - 1) if len(ma) > 0 else [x[0]] * (w - 1)
    return np.array(pad_left.tolist() + ma.tolist())


def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  
    return c * r


def normalize_array(arr: np.ndarray) -> Tuple[np.ndarray, float, float]:
    if arr.size == 0:
        return arr, 0.0, 1.0
    min_v = float(arr.min())
    max_v = float(arr.max())
    if max_v - min_v < 1e-12:
        return np.zeros_like(arr), min_v, max_v
    return (arr - min_v) / (max_v - min_v), min_v, max_v


def softmax(x: np.ndarray, T: float = 1.0) -> np.ndarray:
    x = np.array(x, dtype=float) / max(T, 1e-8)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def parse_required_models(cell: str) -> List[str]:
    if isinstance(cell, list):
        return cell
    try:
        return json.loads(cell)
    except Exception:
        try:
            fixed = cell.replace('""', '"')
            return json.loads(fixed)
        except Exception:
            return [s.strip().strip('"').strip("'") for s in cell.strip('[]').split(',') if s.strip()]


def write_csv_header_if_missing(path: str, header: str):
    if not os.path.exists(path) or os.stat(path).st_size == 0:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(header + "\n")


def generate_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{timestamp_tag()}_{uuid.uuid4().hex[:6]}"

