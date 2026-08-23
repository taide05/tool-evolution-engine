from ..utils.database import get_connection


async def get_db():
    conn = await get_connection()
    try:
        yield conn
    finally:
        await conn.close()
