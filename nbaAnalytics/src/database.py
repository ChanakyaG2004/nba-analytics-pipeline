from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, String, Integer, Float, DateTime, Index
import datetime

# Async driver (asyncpg) for sub-120ms performance
DATABASE_URL = "postgresql+asyncpg://admin:password123@localhost:5432/nba_db"

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

class PlayByPlay(Base):
    __tablename__ = "play_by_play"
    id = Column(Integer, primary_key=True)
    game_id = Column(String(12), index=True)
    event_num = Column(Integer)
    period = Column(Integer)
    pctimestring = Column(String(10))
    score = Column(String(20))
    scoremargin = Column(Integer)
    home_win_prob = Column(Float) # From our model
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    # Database Tuning: Indexing for fast dashboard lookups
    __table_args__ = (Index('idx_game_event', 'game_id', 'event_num'),)