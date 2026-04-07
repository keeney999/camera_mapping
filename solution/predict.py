import argparse
import json
from pathlib import Path
from typing import Tuple

from .model import HomographyMapper


class Predictor:
    def __init__(self, models_dir: str):
        self.models_dir = Path(models_dir)
        self._models = {}
        for cam in ["top", "bottom"]:
            path = self.models_dir / f"homography_{cam}.npy"
            if path.exists():
                self._models[cam] = HomographyMapper.load(path)

    def predict(self, x: float, y: float, source: str) -> Tuple[float, float]:
        if source not in self._models:
            raise ValueError(f"Неизвестный source: {source}, доступны top/bottom")
        pt = self._models[source].predict([[x, y]])[0]
        return float(pt[0]), float(pt[1])


def main():
    parser = argparse.ArgumentParser(description="Предсказание или оценка")
    parser.add_argument(
        "--data_root", type=str, help="Путь к coord_data (нужен для оценки)"
    )
    parser.add_argument("--models_dir", type=str, default="./models")
    parser.add_argument(
        "--eval", action="store_true", help="Оценить модели на валидации"
    )
    args = parser.parse_args()

    if args.eval:
        if not args.data_root:
            parser.error("Для --eval нужно указать --data_root")
        from .eval import evaluate_models

        metrics = evaluate_models(Path(args.data_root), Path(args.models_dir))
        with open(Path(args.models_dir) / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print("Метрики сохранены в", Path(args.models_dir) / "metrics.json")
    else:
        # Интерактивный режим для теста
        pred = Predictor(args.models_dir)
        print("Пример: top 743.96 524.59")
        while True:
            try:
                inp = input("> ").strip()
                if not inp:
                    continue
                src, xs, ys = inp.split()
                x, y = float(xs), float(ys)
                xp, yp = pred.predict(x, y, src)
                print(f"→ door2: ({xp:.1f}, {yp:.1f})")
            except (KeyboardInterrupt, EOFError):
                break
            except Exception as e:
                print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()
