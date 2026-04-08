import cv2
import numpy as np
from pathlib import Path
from typing import Union, Optional
from sklearn.neighbors import NearestNeighbors


class BaseMapper:
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
    def __init__(self, degree: int = 3, use_normalization: bool = True):
        self.degree = degree
        self.use_normalization = use_normalization
        self.coef_x: Optional[np.ndarray] = None
        self.coef_y: Optional[np.ndarray] = None
        self.x_mean = self.x_std = self.y_mean = self.y_std = None

    def _normalize(self, points: np.ndarray, fit: bool = False) -> np.ndarray:
        if not self.use_normalization:
            return points
        if fit:
            self.x_mean = points[:, 0].mean()
            self.x_std = points[:, 0].std()
            self.y_mean = points[:, 1].mean()
            self.y_std = points[:, 1].std()
            if self.x_std == 0:
                self.x_std = 1
            if self.y_std == 0:
                self.y_std = 1
        x_norm = (points[:, 0] - self.x_mean) / self.x_std
        y_norm = (points[:, 1] - self.y_mean) / self.y_std
        return np.column_stack([x_norm, y_norm])

    def _build_design_matrix(self, points: np.ndarray) -> np.ndarray:
        points = self._normalize(points, fit=False)
        x = points[:, 0]
        y = points[:, 1]
        if self.degree == 3:
            return np.column_stack(
                [
                    np.ones_like(x),
                    x,
                    y,
                    x * x,
                    x * y,
                    y * y,
                    x * x * x,
                    x * x * y,
                    x * y * y,
                    y * y * y,
                ]
            )
        else:
            # fallback degree=2
            return np.column_stack([np.ones_like(x), x, y, x * x, x * y, y * y])

    def fit(self, src_points: np.ndarray, dst_points: np.ndarray) -> bool:
        if len(src_points) < 10:
            return False
        # Нормализуем источники
        src_norm = self._normalize(src_points, fit=True)
        X = self._build_design_matrix(src_norm)
        # Регуляризация
        XTX = X.T @ X
        reg = np.eye(XTX.shape[0]) * 0.01
        self.coef_x = np.linalg.solve(XTX + reg, X.T @ dst_points[:, 0])
        self.coef_y = np.linalg.solve(XTX + reg, X.T @ dst_points[:, 1])
        return True

    def predict(self, points: np.ndarray) -> np.ndarray:
        if self.coef_x is None or self.coef_y is None:
            raise ValueError("Модель не обучена")
        pts = np.array(points, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(1, 2)
        src_norm = self._normalize(pts, fit=False)
        X = self._build_design_matrix(src_norm)
        pred_x = X @ self.coef_x
        pred_y = X @ self.coef_y
        result = np.column_stack([pred_x, pred_y])
        if np.array(points).ndim == 1:
            return result[0]
        return result

    def save(self, path: Union[str, Path]) -> None:
        data = {
            "degree": self.degree,
            "use_normalization": self.use_normalization,
            "coef_x": self.coef_x,
            "coef_y": self.coef_y,
            "x_mean": self.x_mean,
            "x_std": self.x_std,
            "y_mean": self.y_mean,
            "y_std": self.y_std,
        }
        np.save(str(path), data, allow_pickle=True)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PolynomialMapper":
        data = np.load(str(path), allow_pickle=True).item()
        obj = cls(degree=data["degree"], use_normalization=data["use_normalization"])
        obj.coef_x = data["coef_x"]
        obj.coef_y = data["coef_y"]
        obj.x_mean = data["x_mean"]
        obj.x_std = data["x_std"]
        obj.y_mean = data["y_mean"]
        obj.y_std = data["y_std"]
        return obj


class KNNMapper(BaseMapper):
    """
    K ближайших соседей (KNN) для маппинга.
    Предсказание: усреднение dst-координат K ближайших src-точек.
    """

    def __init__(self, k: int = 5):
        self.k = k
        self.src_points: Optional[np.ndarray] = None
        self.dst_points: Optional[np.ndarray] = None
        self.nbrs: Optional[NearestNeighbors] = None

    def fit(self, src_points: np.ndarray, dst_points: np.ndarray) -> bool:
        if len(src_points) < self.k:
            return False
        self.src_points = src_points.astype(np.float32)
        self.dst_points = dst_points.astype(np.float32)
        self.nbrs = NearestNeighbors(n_neighbors=self.k, algorithm="auto")
        self.nbrs.fit(self.src_points)
        return True

    def predict(self, points: np.ndarray) -> np.ndarray:
        if self.nbrs is None:
            raise ValueError("Модель не обучена")
        pts = np.array(points, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
        distances, indices = self.nbrs.kneighbors(pts)
        # Усредняем соответствующие dst_points
        pred = np.array([self.dst_points[idx].mean(axis=0) for idx in indices])
        if np.array(points).ndim == 1:
            return pred[0]
        return pred

    def save(self, path: Union[str, Path]) -> None:
        data = {
            "k": self.k,
            "src_points": self.src_points,
            "dst_points": self.dst_points,
        }
        np.save(str(path), data, allow_pickle=True)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "KNNMapper":
        data = np.load(str(path), allow_pickle=True).item()
        obj = cls(k=data["k"])
        obj.src_points = data["src_points"]
        obj.dst_points = data["dst_points"]
        obj.nbrs = NearestNeighbors(n_neighbors=obj.k, algorithm="auto")
        obj.nbrs.fit(obj.src_points)
        return obj
