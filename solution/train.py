import argparse
import logging
from pathlib import Path

from .data_loader import load_split, collect_all_points
from .model import HomographyMapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def main():
    parser = argparse.ArgumentParser(
        description="Обучить гомографии top→door2 и bottom→door2"
    )
    parser.add_argument(
        "--data_root", type=str, required=True, help="Путь к папке coord_data"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./models", help="Куда сохранить .npy файлы"
    )
    parser.add_argument(
        "--ransac_threshold", type=float, default=3.0, help="Порог RANSAC в пикселях"
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_sessions, _ = load_split(data_root)
    logging.info(f"Найдено тренировочных сессий: {len(train_sessions)}")

    for camera in ["top", "bottom"]:
        logging.info(f"Обработка {camera}...")
        src, dst = collect_all_points(data_root, train_sessions, camera)
        if len(src) < 4:
            logging.error(f"Недостаточно точек для {camera} (нужно >=4)")
            continue
        model = HomographyMapper(ransac_threshold=args.ransac_threshold)
        if model.fit(src, dst):
            out_path = output_dir / f"homography_{camera}.npy"
            model.save(out_path)
            logging.info(f"Сохранено в {out_path}")
        else:
            logging.error(f"Не удалось подобрать гомографию для {camera}")


if __name__ == "__main__":
    main()
