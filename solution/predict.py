import argparse
import json
from pathlib import Path
from typing import Tuple, Union

from .model import HomographyMapper, PolynomialMapper


class Predictor:
    def __init__(self, models_dir: Union[str, Path], model_type: str = "polynomial"):
        self.models_dir = Path(models_dir)
        self.model_type = model_type
        self._models = {}
        for cam in ["top", "bottom"]:
            path = self.models_dir / f"{model_type}_{cam}.npy"
            if not path.exists():
                raise FileNotFoundError(
                    f"Модель {model_type}_{cam} не найдена в {path}"
                )
            if model_type == "homography":
                self._models[cam] = HomographyMapper.load(path)
            else:
                self._models[cam] = PolynomialMapper.load(path)

    def predict(self, x: float, y: float, source: str) -> Tuple[float, float]:
        if source not in self._models:
            raise ValueError(f"Неизвестный source: {source}")
        pt = self._models[source].predict([[x, y]])[0]
        return float(pt[0]), float(pt[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root", type=str, help="Путь к coord_data (нужен для оценки)"
    )
    parser.add_argument("--models_dir", type=str, default="./models")
    parser.add_argument(
        "--model_type",
        type=str,
        default="polynomial",
        choices=["homography", "polynomial"],
    )
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    if args.eval:
        if not args.data_root:
            parser.error("--eval требует --data_root")
        from .eval import evaluate_models

        metrics = evaluate_models(
            Path(args.data_root), Path(args.models_dir), args.model_type
        )
        with open(Path(args.models_dir) / f"metrics_{args.model_type}.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Метрики сохранены в models/metrics_{args.model_type}.json")
    else:
        pred = Predictor(args.models_dir, args.model_type)
        print(f"Интерактивный режим (модель: {args.model_type})")
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
