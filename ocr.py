# ocr.py — распознавание номерных знаков

import logging
import easyocr

logger = logging.getLogger(__name__)

# Инициализируется один раз при старте — это долго (5-10 сек), потом быстро
# gpu=False — используем CPU, gpu=True если есть NVIDIA и установлен CUDA
reader = easyocr.Reader(["en"], gpu=False)


def recognize_plate(car_image) -> str | None:
    """
    Принимает: numpy array — вырезанная машина крупным планом
    Возвращает: строку с номером или None если не распознал

    Логика:
    1. EasyOCR читает весь текст с изображения
    2. Фильтруем по длине и уверенности
    3. Берём кандидата с наибольшей уверенностью
    """
    if car_image is None or car_image.size == 0:
        return None

    try:
        results = reader.readtext(car_image)

        candidates = []
        for (bbox, text, confidence) in results:
            # Убираем пробелы, приводим к верхнему регистру
            cleaned = text.strip().upper().replace(" ", "")

            # Казахстанский номер — минимум 5 символов, уверенность > 40%
            if len(cleaned) >= 5 and confidence > 0.4:
                candidates.append((confidence, cleaned))
                logger.debug(f"OCR кандидат: '{cleaned}' conf={confidence:.2f}")

        if not candidates:
            logger.debug("OCR: номер не распознан")
            return None

        # Берём с наибольшей уверенностью
        candidates.sort(reverse=True)
        plate = candidates[0][1]
        logger.info(f"OCR: номер распознан — {plate}")
        return plate

    except Exception as e:
        logger.error(f"OCR ошибка: {e}")
        return None