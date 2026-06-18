import datetime as dt
import os

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker


ASYNC_DATABASE_URL = os.getenv(
    "ASYNC_DATABASE_URL",
    "postgresql+asyncpg://admin:password123@localhost:5432/nba_db",
)

engine = create_async_engine(ASYNC_DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()


class Game(Base):
    __tablename__ = "games"

    game_id = Column(String, primary_key=True)
    game_date = Column(DateTime(timezone=True))
    name = Column(Text)
    short_name = Column(Text)
    status = Column(Text)
    home_team_id = Column(String)
    home_team_name = Column(Text)
    home_team_abbrev = Column(String)
    away_team_id = Column(String)
    away_team_name = Column(Text)
    away_team_abbrev = Column(String)
    updated_at = Column(DateTime(timezone=True), default=dt.datetime.utcnow)


class PlayByPlay(Base):
    __tablename__ = "play_by_play"

    game_id = Column(String, ForeignKey("games.game_id", ondelete="CASCADE"), primary_key=True)
    play_id = Column(String, primary_key=True)
    sequence_number = Column(Integer, nullable=False)
    period = Column(Integer)
    clock_display = Column(String)
    seconds_remaining = Column(Integer)
    home_score = Column(Integer)
    away_score = Column(Integer)
    score_margin = Column(Integer)
    scoring_play = Column(Boolean)
    team_id = Column(String)
    play_type_id = Column(String)
    play_type_text = Column(Text)
    text = Column(Text)
    raw_json = Column(JSONB, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=dt.datetime.utcnow)

    __table_args__ = (
        Index("idx_play_by_play_game_sequence", "game_id", "sequence_number"),
        Index("idx_play_by_play_period", "period"),
    )


class IngestRun(Base):
    __tablename__ = "ingest_runs"

    run_id = Column(UUID(as_uuid=True), primary_key=True)
    started_at = Column(DateTime(timezone=True), nullable=False)
    finished_at = Column(DateTime(timezone=True))
    source = Column(Text, nullable=False)
    requested_events = Column(Integer, nullable=False, default=0)
    successful_events = Column(Integer, nullable=False, default=0)
    total_plays = Column(Integer, nullable=False, default=0)
    status = Column(Text, nullable=False)
    error = Column(Text)
