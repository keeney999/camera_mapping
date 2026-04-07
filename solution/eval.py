import numpy as np
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from .data_loader import load_split, load_correspondences
from .model import HomographyMapper, PolynomialMapper


def compute_med(data_root: Path, sessions: List[str], camera: str, model) -> float:
    errors = []
    for rel_path in tqdm(sessions, desc=f"Evaluating {camera}"):
        session_path = data_root / rel_path
        for src_pts, true_dst in load_correspondences(session_path, camera):
            pred_dst = model.predict(src_pts)
            dists = np.linalg.norm(pred_dst - true_dst, axis=1)
            errors.extend(dists)
    return float(np.mean(errors)) if errors else float("inf")


def evaluate_models(
    data_root: Path, models_dir: Path, model_type: str = "polynomial"
) -> Dict[str, float]:
    _, val_sessions = load_split(data_root)
    metrics = {}
    for camera in ["top", "bottom"]:
        model_path = models_dir / f"{model_type}_{camera}.npy"
        if not model_path.exists():
            print(f"Модель {model_type}_{camera} не найдена")
            metrics[f"{camera}_med"] = float("nan")
            continue
        if model_type == "homography":
            model = HomographyMapper.load(model_path)
        else:
            model = PolynomialMapper.load(model_path)
        med = compute_med(data_root, val_sessions, camera, model)
        metrics[f"{camera}_med"] = med
        print(f"{camera} → door2  MED = {med:.2f} px")
    return metrics
