# config.py — все настройки в одном месте

# Источник видео
# 0 = встроенная камера
# "test_video.mp4" = видеофайл
# "rtsp://192.168.1.10/stream" = IP-камера
VIDEO_SOURCE = "test_video.mp4"

# Y-координата стоп-линии в пикселях
# Подбирается вручную под конкретную камеру/видео
STOP_LINE_Y = 350

# Уверенность YOLO — объекты ниже этого порога игнорируются
CONFIDENCE_THRESHOLD = 0.5

# Классы YOLO (датасет COCO):
# 2=car, 3=motorcycle, 5=bus, 7=truck
VEHICLE_CLASSES = [2, 3, 5, 7]

# База данных
DATABASE_URL = "postgresql://postgres:12345@localhost:5432/sergek"

# Папка для сохранения фото нарушений
VIOLATIONS_DIR = "violations"

# Порт веб-интерфейса
API_PORT = 8002

# Показывать окно OpenCV (True/False)
# False = только веб-интерфейс в браузере
SHOW_WINDOW = True