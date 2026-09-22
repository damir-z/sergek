# detector.py — детекция машин и отслеживание нарушений стоп-линии

import cv2
import os
import logging
from datetime import datetime
from ultralytics import YOLO
from config import (
    STOP_LINE_Y,
    CONFIDENCE_THRESHOLD,
    VEHICLE_CLASSES,
    VIOLATIONS_DIR,
)

logger = logging.getLogger(__name__)

# Загружаем модель один раз при старте
model = YOLO("yolov8n.pt")  # скачается автоматически при первом запуске

# Словарь: track_id → Y-координата нижней границы машины на предыдущем кадре
previous_positions: dict[int, float] = {}

# Множество ID машин которых уже сфотографировали — чтобы не дублировать
already_captured: set[int] = set()


def draw_stop_line(frame):
    """Рисует красную стоп-линию на кадре."""
    h, w = frame.shape[:2]
    cv2.line(frame, (0, STOP_LINE_Y), (w, STOP_LINE_Y), (0, 0, 255), 3)
    cv2.putText(
        frame, "STOP LINE",
        (10, STOP_LINE_Y - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
    )
    return frame


def capture_violation(frame, x1, y1, x2, y2, track_id):
    """
    Сохраняет два фото:
    - full_*.jpg — полный кадр с красной рамкой вокруг нарушителя
    - crop_*.jpg — вырезанная машина крупным планом (для OCR и интерфейса)
    """
    h, w = frame.shape[:2]

    # Отступ вокруг машины — чтобы крупный план был читаемым
    pad = 15
    cx1 = max(0, x1 - pad)
    cy1 = max(0, y1 - pad)
    cx2 = min(w, x2 + pad)
    cy2 = min(h, y2 + pad)

    # Вырезаем машину
    car_crop = frame[cy1:cy2, cx1:cx2].copy()

    timestamp = datetime.now()
    ts_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")

    full_path = os.path.join(VIOLATIONS_DIR, f"full_{track_id}_{ts_str}.jpg")
    crop_path = os.path.join(VIOLATIONS_DIR, f"crop_{track_id}_{ts_str}.jpg")

    # Полный кадр — рисуем красную рамку и подпись нарушителя
    evidence = frame.copy()
    cv2.rectangle(evidence, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.putText(
        evidence, f"VIOLATION  ID:{track_id}",
        (x1, max(y1 - 12, 20)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
    )
    cv2.imwrite(full_path, evidence)
    cv2.imwrite(crop_path, car_crop)

    return car_crop, full_path, crop_path, timestamp


def process_frame(frame):
    """
    Главная функция обработки кадра.

    Принимает: frame (numpy array — один кадр видео)
    Возвращает:
        annotated_frame — кадр с разметкой (рамки, стоп-линия, ID)
        violations      — список словарей с данными о нарушениях
    """
    violations = []

    # Запускаем YOLO с трекингом ByteTrack — он стабильнее держит ID
    results = model.track(
        frame,
        persist=True,
        conf=CONFIDENCE_THRESHOLD,
        tracker="bytetrack.yaml",
        iou=0.5,
        verbose=False,
    )

    # Если на кадре нет объектов с ID — просто рисуем линию и выходим
    if results[0].boxes.id is None:
        return draw_stop_line(frame), violations

    boxes = results[0].boxes

    for i in range(len(boxes)):
        cls = int(boxes.cls[i])

        # Пропускаем всё кроме машин/мотоциклов/автобусов/грузовиков
        if cls not in VEHICLE_CLASSES:
            continue

        track_id = int(boxes.id[i])
        x1, y1, x2, y2 = map(int, boxes.xyxy[i])
        conf = float(boxes.conf[i])

        # Цвет рамки — зелёный по умолчанию
        color = (0, 200, 0)

        # ── Проверка пересечения стоп-линии ──
        if track_id in previous_positions:
            prev_y2 = previous_positions[track_id]

            # Машина пересекла линию: была выше (prev_y2 < STOP_LINE_Y)
            # и теперь ниже (y2 >= STOP_LINE_Y)
            if prev_y2 < STOP_LINE_Y <= y2:
                if track_id not in already_captured:
                    already_captured.add(track_id)
                    color = (0, 0, 255)  # красный — нарушитель

                    car_crop, full_path, crop_path, timestamp = capture_violation(
                        frame, x1, y1, x2, y2, track_id
                    )
                    violations.append({
                        "crop": car_crop,
                        "full_path": full_path,
                        "crop_path": crop_path,
                        "timestamp": timestamp,
                    })
                    logger.info(
                        f"Нарушение зафиксировано: ID={track_id} "
                        f"y={y2} линия={STOP_LINE_Y}"
                    )

        # Запоминаем позицию для следующего кадра
        previous_positions[track_id] = float(y2)

        # Рисуем рамку вокруг машины
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Подпись: ID и уверенность модели
        label = f"ID:{track_id}  {conf:.2f}"
        cv2.putText(
            frame, label,
            (x1, max(y1 - 8, 16)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

    draw_stop_line(frame)
    return frame, violations