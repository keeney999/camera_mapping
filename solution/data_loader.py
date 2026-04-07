import json
from pathlib import Path
from typing import List, Tuple
import numpy as np
from tqdm import tqdm


def load_split(data_root: Path) -> Tuple[List[str], List[str]]:
    """Прочитать split.json, вернуть (train_paths, val_paths)."""
    with open(data_root / "split.json") as f:
        split = json.load(f)
    return split["train"], split["val"]


def load_correspondences(
    session_path: Path, camera: str
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Для одной сессии и камеры ('top' или 'bottom') загрузить все пары точек.
    Каждый элемент списка: (src_pts, dst_pts) – массивы (N,2).
    src_pts – координаты на камере top/bottom, dst_pts – на door2.
    """
    coords_file = session_path / f"coords_{camera}.json"
    if not coords_file.exists():
        return []

    with open(coords_file) as f:
        data = json.load(f)  # список объектов

    pairs = []
    for item in data:
        door2 = {p["number"]: (p["x"], p["y"]) for p in item["image1_coordinates"]}
        source = {p["number"]: (p["x"], p["y"]) for p in item["image2_coordinates"]}
        common = set(door2.keys()) & set(source.keys())
        if not common:
            continue
        src = np.array([source[n] for n in common], dtype=np.float32)
        dst = np.array([door2[n] for n in common], dtype=np.float32)
        pairs.append((src, dst))
    return pairs


def collect_all_points(
    data_root: Path, sessions: List[str], camera: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Собрать все точки из всех сессий сплита для одной камеры."""
    all_src, all_dst = [], []
    for rel_path in tqdm(sessions, desc=f"Loading {camera}"):
        session_path = data_root / rel_path
        for src, dst in load_correspondences(session_path, camera):
            all_src.append(src)
            all_dst.append(dst)
    if not all_src:
        return np.empty((0, 2)), np.empty((0, 2))
    return np.vstack(all_src), np.vstack(all_dst)
