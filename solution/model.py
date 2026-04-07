import cv2
import numpy as np
from pathlib import Path
from typing import Union, Optional


class BaseMapper:
    """Абстрактный базовый класс для всех мапперов."""

    def fit(self, src: np.ndarray, dst: np.ndarray) -> bool:
        raise NotImplementedError

    def predict(self, points: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def save(self, path: Union[str, Path]) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BaseMapper":
        raise NotImplementedError


class HomographyMapper(BaseMapper):
    """Гомографическое преобразование (исходный вариант)."""

    def __init__(self, ransac_threshold: float = 3.0):
        self.H: Optional[np.ndarray] = None
        self.ransac_threshold = ransac_threshold

    def fit(self, src_points: np.ndarray, dst_points: np.ndarray) -> bool:
        if len(src_points) < 4:
            return False
        H, _ = cv2.findHomography(
            src_points,
            dst_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold,
        )
        if H is None:
            return False
        self.H = H
        return True

    def predict(self, points: np.ndarray) -> np.ndarray:
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

    def save(self, path: Union[str, Path]) -> None:
        np.save(str(path), self.H)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "HomographyMapper":
        obj = cls()
        obj.H = np.load(str(path))
        return obj


class PolynomialMapper(BaseMapper):
    """
    Полиномиальная регрессия 2-го порядка.
    Для каждой выходной координаты (x', y') обучается отдельная модель:
        x' = a0 + a1*x + a2*y + a3*x*y + a4*x^2 + a5*y^2
        y' = b0 + b1*x + b2*y + b3*x*y + b4*x^2 + b5*y^2
    """

    def __init__(self, degree: int = 2):
        self.degree = degree
        self.coef_x: Optional[np.ndarray] = None
        self.coef_y: Optional[np.ndarray] = None

    def _build_design_matrix(self, points: np.ndarray) -> np.ndarray:
        """Строит матрицу признаков: [1, x, y, xy, x^2, y^2] для degree=2."""
        x = points[:, 0]
        y = points[:, 1]
        if self.degree == 2:
            return np.column_stack([np.ones_like(x), x, y, x * y, x * x, y * y])
        else:
            raise NotImplementedError("Только степень 2")

    def fit(self, src_points: np.ndarray, dst_points: np.ndarray) -> bool:
        if len(src_points) < 6:  # нужно минимум 6 точек для 6 коэффициентов
            return False
        X = self._build_design_matrix(src_points)
        # Решаем методом наименьших квадратов
        self.coef_x, _, _, _ = np.linalg.lstsq(X, dst_points[:, 0], rcond=None)
        self.coef_y, _, _, _ = np.linalg.lstsq(X, dst_points[:, 1], rcond=None)
        return True

    def predict(self, points: np.ndarray) -> np.ndarray:
        if self.coef_x is None or self.coef_y is None:
            raise ValueError("Модель не обучена")
        pts = np.array(points, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(1, 2)
        X = self._build_design_matrix(pts)
        pred_x = X @ self.coef_x
        pred_y = X @ self.coef_y
        result = np.column_stack([pred_x, pred_y])
        if np.array(points).ndim == 1:
            return result[0]
        return result

    def save(self, path: Union[str, Path]) -> None:
        data = {"degree": self.degree, "coef_x": self.coef_x, "coef_y": self.coef_y}
        np.save(str(path), data, allow_pickle=True)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PolynomialMapper":
        data = np.load(str(path), allow_pickle=True).item()
        obj = cls(degree=data["degree"])
        obj.coef_x = data["coef_x"]
        obj.coef_y = data["coef_y"]
        return obj
