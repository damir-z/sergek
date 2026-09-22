# api.py — FastAPI веб-интерфейс
# Отдельный файл, не смешиваем с логикой камеры

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from db import get_violations, search_by_plate, update_status, get_stats

app = FastAPI(title="Sergek API")

# Отдаём фотографии нарушений по URL /photos/имя_файла.jpg
app.mount("/photos", StaticFiles(directory="violations"), name="photos")

# Отдаём статические файлы (index.html, стили и т.д.)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Видеострим ────────────────────────────────────────────────
# Глобальная переменная — текущий кадр в формате JPEG байты
# Заполняется из main.py через set_current_frame()
_current_frame_jpg: bytes | None = None


def set_current_frame(jpg_bytes: bytes):
    """Вызывается из main.py чтобы обновить текущий кадр."""
    global _current_frame_jpg
    _current_frame_jpg = jpg_bytes


def _frame_generator():
    """
    Генератор для MJPEG стриминга.
    Браузер получает поток кадров и показывает как живое видео.
    """
    import time
    while True:
        if _current_frame_jpg:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + _current_frame_jpg
                + b"\r\n"
            )
        time.sleep(0.033)  # ~30 FPS


@app.get("/video")
def video_feed():
    """Эндпоинт видеострима — подключается тегом <img src='/video'>."""
    return StreamingResponse(
        _frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── Страница ──────────────────────────────────────────────────
@app.get("/")
def index():
    return FileResponse("static/index.html")


# ── REST API ──────────────────────────────────────────────────
@app.get("/api/violations")
async def list_violations(limit: int = 50, offset: int = 0):
    return await get_violations(limit, offset)


@app.get("/api/search")
async def search(plate: str):
    return await search_by_plate(plate)


@app.get("/api/stats")
async def stats():
    return await get_stats()


class StatusUpdate(BaseModel):
    status: str  # new / warned / fined


@app.put("/api/violations/{violation_id}")
async def update(violation_id: int, body: StatusUpdate):
    if body.status not in ("new", "warned", "fined"):
        raise HTTPException(
            status_code=400,
            detail="Статус должен быть: new, warned или fined",
        )
    await update_status(violation_id, body.status)
    return {"ok": True}