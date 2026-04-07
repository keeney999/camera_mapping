import cv2
import numpy as np
from typing import Optional, Union


class HomographyMapper:
    """Гомография для отображения точек с одной камеры на другую."""

    def __init__(self, ransac_threshold: float = 3.0):
        self.H: Optional[np.ndarray] = None
        self.ransac_threshold = ransac_threshold

    def fit(self, src_points: np.ndarray, dst_points: np.ndarray) -> bool:
        """Оценить матрицу гомографии по парам точек (N,2)."""
        if len(src_points) < 4:
            return False
        H = cv2.findHomography(
            src_points,
            dst_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold,
        )
        if H is None:
            return False
        self.H = H
        return True

    def predict(self, points: Union[np.ndarray, list]) -> np.ndarray:
        """Применить гомографию к точкам. Вход: (N,2) или (2,)."""
        if self.H is None:
            raise ValueError("Модель не обучена")
        pts = np.array(points, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(1, 1, 2)
        elif pts.ndim == 2 and pts.shape[1] == 2:
            pts = pts.reshape(-1, 1, 2)
        else:
            pts = pts.reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, self.H)
        result = transformed.reshape(-1, 2)
        if np.array(points).ndim == 1:
            return result[0]
        return result

    def save(self, path: str) -> None:
        np.save(path, self.H)

    @classmethod
    def load(cls, path: str) -> "HomographyMapper":
        obj = cls()
        obj.H = np.load(path)
        return obj
