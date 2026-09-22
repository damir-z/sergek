# db.py — работа с базой данных PostgreSQL

import asyncpg
import logging
from config import DATABASE_URL

logger = logging.getLogger(__name__)

_pool = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(DATABASE_URL)
    return _pool


async def init_db():
    """Создаёт таблицу нарушений если не существует."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS violations (
                id          SERIAL PRIMARY KEY,
                plate       VARCHAR(20),
                detected_at TIMESTAMP NOT NULL,
                full_photo  VARCHAR(500),
                crop_photo  VARCHAR(500),
                status      VARCHAR(20) DEFAULT 'new',
                created_at  TIMESTAMP DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_violations_plate
            ON violations(plate)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_violations_detected_at
            ON violations(detected_at DESC)
        """)
    logger.info("БД инициализирована")


async def save_violation(
    plate: str | None,
    timestamp,
    full_path: str,
    crop_path: str,
) -> int | None:
    """Сохраняет нарушение, возвращает ID записи."""
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO violations
                    (plate, detected_at, full_photo, crop_photo, status)
                VALUES ($1, $2, $3, $4, 'new')
                RETURNING id
            """, plate, timestamp, full_path, crop_path)
            vid = row["id"]
            logger.info(f"Нарушение сохранено: ID={vid} номер={plate}")
            return vid
    except Exception as e:
        logger.error(f"save_violation ошибка: {e}")
        return None


async def get_violations(limit: int = 50, offset: int = 0) -> list[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT * FROM violations
            ORDER BY detected_at DESC
            LIMIT $1 OFFSET $2
        """, limit, offset)
        return [dict(r) for r in rows]


async def search_by_plate(plate: str) -> list[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT * FROM violations
            WHERE plate ILIKE $1
            ORDER BY detected_at DESC
        """, f"%{plate}%")
        return [dict(r) for r in rows]


async def update_status(violation_id: int, status: str):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            UPDATE violations SET status = $1 WHERE id = $2
        """, status, violation_id)


async def get_stats() -> dict:
    pool = await get_pool()
    async with pool.acquire() as conn:
        total = await conn.fetchval("SELECT COUNT(*) FROM violations")
        today = await conn.fetchval("""
            SELECT COUNT(*) FROM violations
            WHERE detected_at::date = CURRENT_DATE
        """)
        top_plates = await conn.fetch("""
            SELECT plate, COUNT(*) as cnt
            FROM violations
            WHERE plate IS NOT NULL
            GROUP BY plate
            ORDER BY cnt DESC
            LIMIT 5
        """)
        return {
            "total": total or 0,
            "today": today or 0,
            "top_plates": [dict(r) for r in top_plates],
        }