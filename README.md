# Camera Mapping: top/bottom → door2

Маппинг пиксельных координат с камер top/bottom на кадр door2 с помощью полиномиальной регрессии 2-й степени.

## Стек

- Python 3.10+
- Poetry
- OpenCV (гомография – опционально)
- NumPy, tqdm

## Быстрый старт

```bash
# Клонировать и установить зависимости
poetry install
poetry shell
Обучение (полиномиальная модель по умолчанию)
bash
python -m solution.train --data_root /путь/к/test-task --output_dir ./models
Файлы моделей: polynomial_top.npy, polynomial_bottom.npy.

Оценка на валидации
bash
python -m solution.predict --data_root /путь/к/test-task --models_dir ./models --eval
Результат (MED в пикселях) выводится в консоль и сохраняется в models/metrics_polynomial.json.

Использование в коде
python
from solution.predict import Predictor

p = Predictor("./models")          # по умолчанию polynomial
x_door, y_door = p.predict(743.96, 524.59, "top")
print(x_door, y_door)
Примечания
Если качество низкое, попробуйте модель гомографии: --model_type homography

Для улучшения точности требуется более сложный подход (SIFT + локальные гомографии)

Датасет должен лежать в папке с подпапками train/, val/ и файлом split.json
