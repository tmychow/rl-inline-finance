import sqlite3
from datetime import datetime
from typing import List, Dict, Optional
import asyncio
import aiosqlite
from contextlib import asynccontextmanager

DATABASE_PATH = "financial_data.db"

async def init_database():
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute('''
            CREATE TABLE IF NOT EXISTS tickers (
                ticker TEXT PRIMARY KEY,
                company_name TEXT,
                sector TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        await db.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                metric_name TEXT PRIMARY KEY,
                description TEXT,
                unit TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        await db.execute('''
            CREATE TABLE IF NOT EXISTS financial_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                period TEXT NOT NULL,
                value REAL NOT NULL,
                unit TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (ticker) REFERENCES tickers(ticker),
                FOREIGN KEY (metric_name) REFERENCES metrics(metric_name),
                UNIQUE(ticker, metric_name, period)
            )
        ''')
        
        await db.execute('''
            CREATE INDEX IF NOT EXISTS idx_financial_data_ticker 
            ON financial_data(ticker)
        ''')
        
        await db.execute('''
            CREATE INDEX IF NOT EXISTS idx_financial_data_metric 
            ON financial_data(metric_name)
        ''')
        
        await db.execute('''
            CREATE INDEX IF NOT EXISTS idx_financial_data_period 
            ON financial_data(period)
        ''')
        
        await db.commit()

@asynccontextmanager
async def get_db():
    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        yield db

async def insert_ticker(ticker: str, company_name: str, sector: str = None):
    async with get_db() as db:
        await db.execute(
            "INSERT OR REPLACE INTO tickers (ticker, company_name, sector) VALUES (?, ?, ?)",
            (ticker, company_name, sector)
        )
        await db.commit()

async def insert_metric(metric_name: str, description: str, unit: str):
    async with get_db() as db:
        await db.execute(
            "INSERT OR REPLACE INTO metrics (metric_name, description, unit) VALUES (?, ?, ?)",
            (metric_name, description, unit)
        )
        await db.commit()

async def insert_financial_data(ticker: str, metric_name: str, period: str, value: float, unit: str = None):
    async with get_db() as db:
        await db.execute(
            "INSERT OR REPLACE INTO financial_data (ticker, metric_name, period, value, unit) VALUES (?, ?, ?, ?, ?)",
            (ticker, metric_name, period, value, unit)
        )
        await db.commit()

async def get_all_tickers() -> List[str]:
    async with get_db() as db:
        async with db.execute("SELECT ticker FROM tickers ORDER BY ticker") as cursor:
            rows = await cursor.fetchall()
            return [row["ticker"] for row in rows]

async def get_all_metrics() -> List[Dict[str, str]]:
    async with get_db() as db:
        async with db.execute("SELECT metric_name, description, unit FROM metrics ORDER BY metric_name") as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

async def get_financial_value(ticker: str, metric_name: str, period: str) -> Optional[Dict[str, any]]:
    async with get_db() as db:
        async with db.execute(
            "SELECT value, unit FROM financial_data WHERE ticker = ? AND metric_name = ? AND period = ?",
            (ticker, metric_name, period)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return {"value": row["value"], "unit": row["unit"]}
            return None

async def get_available_periods(ticker: str, metric_name: str) -> List[str]:
    async with get_db() as db:
        async with db.execute(
            "SELECT DISTINCT period FROM financial_data WHERE ticker = ? AND metric_name = ? ORDER BY period DESC",
            (ticker, metric_name)
        ) as cursor:
            rows = await cursor.fetchall()
            return [row["period"] for row in rows]

def parse_period(period: str) -> tuple:
    """Parse a period string like '2024Q1' or '2024FY' into (year, quarter)"""
    if 'FY' in period:
        year = int(period.replace('FY', ''))
        return (year, 4)  # FY is like Q4 for sorting
    elif 'Q' in period:
        parts = period.split('Q')
        year = int(parts[0])
        quarter = int(parts[1])
        return (year, quarter)
    return (0, 0)

async def get_latest_period(ticker: str, metric_name: str) -> Optional[str]:
    """Get the most recent period for a given ticker and metric"""
    periods = await get_available_periods(ticker, metric_name)
    if not periods:
        return None
    
    # Sort periods properly (2024Q2 > 2024Q1, 2024FY > 2024Q4)
    sorted_periods = sorted(periods, key=parse_period, reverse=True)
    return sorted_periods[0] if sorted_periods else None


if __name__ == "__main__":
    asyncio.run(init_database())
    print(f"Database initialized at {DATABASE_PATH}")