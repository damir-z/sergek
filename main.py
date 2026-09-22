# main.py — точка входа
# Запускает:
# 1. FastAPI сервер в отдельном потоке (api.py)
# 2. Цикл обработки камеры в основном потоке (asyncio)

import asyncio
import logging
import threading
import cv2
import uvicorn

from config import VIDEO_SOURCE, SHOW_WINDOW, API_PORT
from detector import process_frame
from ocr import recognize_plate
from db import init_db, save_violation
from api import app, set_current_frame

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


async def handle_violation(violation: dict):
    """
    Обрабатывает одно нарушение:
    1. Распознаёт номер через OCR
    2. Сохраняет в БД
    3. Выводит в консоль
    """
    print("\n" + "=" * 55)
    print("🚨  НАРУШЕНИЕ ЗАФИКСИРОВАНО")
    print(f"    Время:  {violation['timestamp'].strftime('%d.%m.%Y %H:%M:%S')}")
    print(f"    Фото:   {violation['full_path']}")
    print("    Распознаю номер...")

    plate = recognize_plate(violation["crop"])

    if plate:
        print(f"    Номер:  {plate}")
    else:
        print("    Номер:  не распознан")

    vid = await save_violation(
        plate=plate,
        timestamp=violation["timestamp"],
        full_path=violation["full_path"],
        crop_path=violation["crop_path"],
    )
    print(f"    БД ID:  {vid}")
    print("=" * 55 + "\n")


async def camera_loop():
    """
    Основной цикл:
    - Читает кадры из видео/камеры
    - Передаёт в детектор
    - Обрабатывает нарушения
    - Обновляет кадр для веб-стриминга
    - Показывает окно OpenCV если SHOW_WINDOW=True
    """
    await init_db()

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        logger.error(f"Не удалось открыть источник: {VIDEO_SOURCE}")
        return

    logger.info(f"Источник открыт: {VIDEO_SOURCE}")
    logger.info(f"Веб-интерфейс:   http://localhost:{API_PORT}")
    logger.info("Нажми Q в окне OpenCV для выхода")

    while True:
        ret, frame = cap.read()

        if not ret:
            # Видеофайл закончился — перематываем в начало
            logger.info("Конец видео — перематываю...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Детектируем машины и проверяем стоп-линию
        annotated_frame, violations = process_frame(frame)

        # Кодируем кадр в JPEG и отправляем в веб-стрим
        _, buf = cv2.imencode(
            ".jpg", annotated_frame,
            [cv2.IMWRITE_JPEG_QUALITY, 75]
        )
        set_current_frame(buf.tobytes())

        # Обрабатываем нарушения (OCR + БД)
        for v in violations:
            await handle_violation(v)

        # Показываем окно OpenCV если включено
        if SHOW_WINDOW:
            cv2.imshow("Sergek  —  Q для выхода", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Небольшая пауза чтобы не грузить CPU на 100%
        await asyncio.sleep(0.01)

    cap.release()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()
    logger.info("Камера остановлена")


def start_api():
    """Запускает FastAPI в отдельном потоке."""
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=API_PORT,
        log_level="warning",
    )


if __name__ == "__main__":
    # Запускаем API в фоновом потоке
    api_thread = threading.Thread(target=start_api, daemon=True)
    api_thread.start()
    logger.info(f"API запущен на порту {API_PORT}")

    # Запускаем цикл камеры в основном потоке
    asyncio.run(camera_loop())