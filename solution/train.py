import argparse
import logging
from pathlib import Path

from .data_loader import load_split, collect_all_points
from .model import HomographyMapper, PolynomialMapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./models")
    parser.add_argument(
        "--model_type",
        type=str,
        default="polynomial",
        choices=["homography", "polynomial"],
    )
    parser.add_argument("--ransac_threshold", type=float, default=3.0)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_sessions, _ = load_split(data_root)
    logging.info(f"Тренировочных сессий: {len(train_sessions)}")

    for camera in ["top", "bottom"]:
        logging.info(f"Обработка {camera}...")
        src, dst = collect_all_points(data_root, train_sessions, camera)
        if len(src) < 6:
            logging.error(f"Недостаточно точек для {camera} (нужно >=6)")
            continue

        if args.model_type == "homography":
            model = HomographyMapper(ransac_threshold=args.ransac_threshold)
        else:
            model = PolynomialMapper(degree=2)

        if model.fit(src, dst):
            out_path = output_dir / f"{args.model_type}_{camera}.npy"
            model.save(out_path)
            logging.info(f"Сохранено в {out_path}")
        else:
            logging.error(f"Не удалось обучить модель для {camera}")


if __name__ == "__main__":
    main()
