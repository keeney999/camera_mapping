# Camera Mapping: top/bottom → door2

Маппинг пиксельных координат с верхней/нижней камеры на основную камеру `door2` с помощью гомографии.

## Стек

- Python 3.10+
- Poetry — управление зависимостями
- OpenCV — оценка гомографии (RANSAC)
- NumPy, tqdm

## Быстрый старт

```bash
# 1. Клонировать проект и перейти в папку
cd test

# 2. Установить зависимости через Poetry
poetry install

# 3. Активировать окружение
poetry shell
Обучение модели
bash
python -m solution.train --data_root /путь/к/test_task --output_dir ./models
После обучения в папке models/ появятся файлы homography_top.npy и homography_bottom.npy.

Оценка на валидации (MED в пикселях)
bash
python -m solution.predict --data_root /путь/к/test_task --models_dir ./models --eval
Результат выводится в консоль и сохраняется в models/metrics.json.

Использование в коде
python
from solution.predict import Predictor

predictor = Predictor("./models")
x_door, y_door = predictor.predict(x=743.96, y=524.59, source="top")
print(x_door, y_door)
Формат метрик
models/metrics.json:

json
{
  "top_med": 4.23,
  "bottom_med": 5.17
}
Примечания
Датасет распакуйте в любую папку и укажите её в --data_root.

Внутри датасета должна быть структура coord_data/, описанная в TASK.md.
