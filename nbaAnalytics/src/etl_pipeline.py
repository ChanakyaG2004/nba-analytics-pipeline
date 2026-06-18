import argparse
import datetime as dt
import json
import os
import uuid

import requests
from prefect import flow, task
from sqlalchemy import create_engine, text


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://admin:password123@localhost:5432/nba_db",
)
ESPN_BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
ENGINE = create_engine(DATABASE_URL, future=True)


CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,
    game_date TIMESTAMPTZ,
    name TEXT,
    short_name TEXT,
    status TEXT,
    home_team_id TEXT,
    home_team_name TEXT,
    home_team_abbrev TEXT,
    away_team_id TEXT,
    away_team_name TEXT,
    away_team_abbrev TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS play_by_play (
    game_id TEXT NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
    play_id TEXT NOT NULL,
    sequence_number INTEGER NOT NULL,
    period INTEGER,
    clock_display TEXT,
    seconds_remaining INTEGER,
    home_score INTEGER,
    away_score INTEGER,
    score_margin INTEGER,
    scoring_play BOOLEAN,
    team_id TEXT,
    play_type_id TEXT,
    play_type_text TEXT,
    text TEXT,
    raw_json JSONB NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (game_id, play_id)
);

CREATE INDEX IF NOT EXISTS idx_play_by_play_game_sequence
    ON play_by_play (game_id, sequence_number);

CREATE INDEX IF NOT EXISTS idx_play_by_play_period
    ON play_by_play (period);

CREATE TABLE IF NOT EXISTS ingest_runs (
    run_id UUID PRIMARY KEY,
    started_at TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ,
    source TEXT NOT NULL,
    requested_events INTEGER NOT NULL DEFAULT 0,
    successful_events INTEGER NOT NULL DEFAULT 0,
    total_plays INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    error TEXT
);
"""


def parse_date(value):
    return dt.datetime.strptime(value, "%Y-%m-%d").date()


def date_range(start_date, end_date):
    current = start_date
    while current <= end_date:
        yield current
        current += dt.timedelta(days=1)


def parse_clock_seconds(clock_display, period):
    if not clock_display or ":" not in clock_display or period is None:
        return None

    try:
        minutes, seconds = clock_display.split(":", 1)
        elapsed_period_seconds = (12 * 60) - ((int(minutes) * 60) + int(seconds))
        regulation_periods_remaining = max(4 - int(period), 0)
        return regulation_periods_remaining * 12 * 60 + max(0, 12 * 60 - elapsed_period_seconds)
    except ValueError:
        return None


@task
def ensure_schema():
    with ENGINE.begin() as conn:
        existing_play_columns = {
            row[0]
            for row in conn.execute(
                text(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name = 'play_by_play'
                    """
                )
            )
        }

        if existing_play_columns and {"play_id", "sequence_number"} - existing_play_columns:
            legacy_table_name = f"play_by_play_legacy_{dt.datetime.utcnow():%Y%m%d%H%M%S}"
            conn.execute(text(f"ALTER TABLE play_by_play RENAME TO {legacy_table_name}"))

        for statement in CREATE_TABLES_SQL.split(";"):
            statement = statement.strip()
            if statement:
                conn.execute(text(statement))


@task(retries=3, retry_delay_seconds=5)
def fetch_scoreboard_event_ids(game_date):
    params = {"dates": game_date.strftime("%Y%m%d")}
    response = requests.get(f"{ESPN_BASE_URL}/scoreboard", params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    return [event["id"] for event in data.get("events", [])]


@task(retries=3, retry_delay_seconds=5)
def fetch_event_summary(event_id):
    response = requests.get(f"{ESPN_BASE_URL}/summary", params={"event": event_id}, timeout=30)
    response.raise_for_status()
    return response.json()


def extract_team(event, home_away):
    competitions = event.get("competitions") or []
    competitors = competitions[0].get("competitors", []) if competitions else []

    for competitor in competitors:
        if competitor.get("homeAway") == home_away:
            team = competitor.get("team", {})
            return {
                "id": team.get("id"),
                "name": team.get("displayName") or team.get("name"),
                "abbrev": team.get("abbreviation"),
            }

    return {"id": None, "name": None, "abbrev": None}


def normalize_game(summary, event_id):
    header = summary.get("header", {})
    competitions = header.get("competitions") or []
    event = competitions[0] if competitions else header
    home = extract_team(header, "home")
    away = extract_team(header, "away")
    fallback_short_name = None

    if away["abbrev"] and home["abbrev"]:
        fallback_short_name = f"{away['abbrev']} @ {home['abbrev']}"

    return {
        "game_id": str(event.get("id") or event_id),
        "game_date": event.get("date") or header.get("date"),
        "name": header.get("name") or fallback_short_name,
        "short_name": header.get("shortName") or fallback_short_name,
        "status": ((event.get("status") or {}).get("type") or {}).get("name"),
        "home_team_id": home["id"],
        "home_team_name": home["name"],
        "home_team_abbrev": home["abbrev"],
        "away_team_id": away["id"],
        "away_team_name": away["name"],
        "away_team_abbrev": away["abbrev"],
    }


def normalize_play(game_id, play, sequence_number):
    period = play.get("period")
    clock = play.get("clock")
    play_type = play.get("type") or {}
    team = play.get("team") or {}
    home_score = play.get("homeScore")
    away_score = play.get("awayScore")
    clock_display = clock.get("displayValue") if isinstance(clock, dict) else clock
    period_number = period.get("number") if isinstance(period, dict) else period

    try:
        score_margin = int(home_score) - int(away_score)
    except (TypeError, ValueError):
        score_margin = None

    return {
        "game_id": str(game_id),
        "play_id": str(play.get("id") or f"{game_id}-{sequence_number}"),
        "sequence_number": sequence_number,
        "period": period_number,
        "clock_display": clock_display,
        "seconds_remaining": parse_clock_seconds(clock_display, period_number),
        "home_score": home_score,
        "away_score": away_score,
        "score_margin": score_margin,
        "scoring_play": play.get("scoringPlay"),
        "team_id": team.get("id"),
        "play_type_id": str(play_type.get("id")) if play_type.get("id") is not None else None,
        "play_type_text": play_type.get("text"),
        "text": play.get("text"),
        "raw_json": play,
    }


@task
def normalize_event(summary, event_id):
    game = normalize_game(summary, event_id)
    plays = [
        normalize_play(game["game_id"], play, sequence_number)
        for sequence_number, play in enumerate(summary.get("plays", []), start=1)
    ]
    return {"game": game, "plays": plays}


@task
def upsert_event(normalized):
    game = normalized["game"]
    plays = normalized["plays"]

    with ENGINE.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO games (
                    game_id, game_date, name, short_name, status,
                    home_team_id, home_team_name, home_team_abbrev,
                    away_team_id, away_team_name, away_team_abbrev
                )
                VALUES (
                    :game_id, :game_date, :name, :short_name, :status,
                    :home_team_id, :home_team_name, :home_team_abbrev,
                    :away_team_id, :away_team_name, :away_team_abbrev
                )
                ON CONFLICT (game_id) DO UPDATE SET
                    game_date = EXCLUDED.game_date,
                    name = EXCLUDED.name,
                    short_name = EXCLUDED.short_name,
                    status = EXCLUDED.status,
                    home_team_id = EXCLUDED.home_team_id,
                    home_team_name = EXCLUDED.home_team_name,
                    home_team_abbrev = EXCLUDED.home_team_abbrev,
                    away_team_id = EXCLUDED.away_team_id,
                    away_team_name = EXCLUDED.away_team_name,
                    away_team_abbrev = EXCLUDED.away_team_abbrev,
                    updated_at = NOW()
                """
            ),
            game,
        )

        for play in plays:
            conn.execute(
                text(
                    """
                    INSERT INTO play_by_play (
                        game_id, play_id, sequence_number, period, clock_display,
                        seconds_remaining, home_score, away_score, score_margin,
                        scoring_play, team_id, play_type_id, play_type_text, text, raw_json
                    )
                    VALUES (
                        :game_id, :play_id, :sequence_number, :period, :clock_display,
                        :seconds_remaining, :home_score, :away_score, :score_margin,
                        :scoring_play, :team_id, :play_type_id, :play_type_text, :text,
                        CAST(:raw_json AS JSONB)
                    )
                    ON CONFLICT (game_id, play_id) DO UPDATE SET
                        sequence_number = EXCLUDED.sequence_number,
                        period = EXCLUDED.period,
                        clock_display = EXCLUDED.clock_display,
                        seconds_remaining = EXCLUDED.seconds_remaining,
                        home_score = EXCLUDED.home_score,
                        away_score = EXCLUDED.away_score,
                        score_margin = EXCLUDED.score_margin,
                        scoring_play = EXCLUDED.scoring_play,
                        team_id = EXCLUDED.team_id,
                        play_type_id = EXCLUDED.play_type_id,
                        play_type_text = EXCLUDED.play_type_text,
                        text = EXCLUDED.text,
                        raw_json = EXCLUDED.raw_json,
                        updated_at = NOW()
                    """
                ),
                {**play, "raw_json": json.dumps(play["raw_json"])},
            )

    return {"game_id": game["game_id"], "plays": len(plays)}


@task
def create_ingest_run(run_id, source, requested_events):
    with ENGINE.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO ingest_runs (
                    run_id, started_at, source, requested_events, status
                )
                VALUES (:run_id, NOW(), :source, :requested_events, 'running')
                """
            ),
            {"run_id": run_id, "source": source, "requested_events": requested_events},
        )


@task
def finish_ingest_run(run_id, successful_events, total_plays, status, error=None):
    with ENGINE.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE ingest_runs
                SET finished_at = NOW(),
                    successful_events = :successful_events,
                    total_plays = :total_plays,
                    status = :status,
                    error = :error
                WHERE run_id = :run_id
                """
            ),
            {
                "run_id": run_id,
                "successful_events": successful_events,
                "total_plays": total_plays,
                "status": status,
                "error": error,
            },
        )


@flow(name="nba-play-by-play-etl")
def ingest_play_by_play(event_ids=None, start_date=None, end_date=None):
    ensure_schema()

    event_ids = list(event_ids or [])
    source_parts = []

    if start_date:
        start = parse_date(start_date) if isinstance(start_date, str) else start_date
        end = parse_date(end_date) if isinstance(end_date, str) else end_date or start
        for game_date in date_range(start, end):
            source_parts.append(game_date.isoformat())
            event_ids.extend(fetch_scoreboard_event_ids(game_date))

    if not event_ids:
        today = dt.date.today()
        source_parts.append(today.isoformat())
        event_ids.extend(fetch_scoreboard_event_ids(today))

    event_ids = sorted(set(str(event_id) for event_id in event_ids))
    run_id = str(uuid.uuid4())
    create_ingest_run(run_id, ",".join(source_parts) or "event_ids", len(event_ids))

    successful_events = 0
    total_plays = 0

    try:
        for event_id in event_ids:
            summary = fetch_event_summary(event_id)
            normalized = normalize_event(summary, event_id)
            result = upsert_event(normalized)
            successful_events += 1
            total_plays += result["plays"]

        finish_ingest_run(run_id, successful_events, total_plays, "success")
    except Exception as exc:
        finish_ingest_run(run_id, successful_events, total_plays, "failed", str(exc))
        raise

    return {
        "run_id": run_id,
        "requested_events": len(event_ids),
        "successful_events": successful_events,
        "total_plays": total_plays,
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Ingest NBA ESPN play-by-play data.")
    parser.add_argument(
        "--event-id",
        action="append",
        dest="event_ids",
        help="ESPN event id to ingest. Repeat for multiple games.",
    )
    parser.add_argument("--start-date", help="Start date for scoreboard backfill, YYYY-MM-DD.")
    parser.add_argument("--end-date", help="End date for scoreboard backfill, YYYY-MM-DD.")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    result = ingest_play_by_play(
        event_ids=args.event_ids,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print(result)
