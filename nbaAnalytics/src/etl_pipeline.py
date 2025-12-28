import requests
import pandas as pd
from sqlalchemy import create_engine, text

ENGINE = create_engine("postgresql://admin:password123@localhost:5432/nba_db")

def get_data():
    # 1. Use a confirmed Game ID (Lakers vs Warriors or similar recent game)
    # ESPN Game IDs for late Dec 2025: Try 401705141 or 401705142
    event_id = "401705141" 
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={event_id}"
    
    print(f"📡 Fetching Game {event_id}...")
    r = requests.get(url)
    data = r.json()
    
    plays = data.get('plays', [])
    if not plays:
        print("❌ This Game ID has no plays yet. Trying a backup ID...")
        # Backup: Fetching the first live/completed game from the scoreboard
        sb = requests.get("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard").json()
        event_id = sb['events'][0]['id']
        data = requests.get(f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={event_id}").json()
        plays = data.get('plays', [])

    # 2. Process and Force Lowercase (Postgres Requirement)
    df = pd.DataFrame(plays)
    df['game_id'] = event_id
    df['period_num'] = df['period'].apply(lambda x: x.get('number') if isinstance(x, dict) else 1)
    df['clock_display'] = df['clock'].apply(lambda x: x.get('displayValue') if isinstance(x, dict) else "0:00")
    
    # Keep only simple columns to avoid errors
    final_df = df[['game_id', 'period_num', 'clock_display', 'text']].copy()
    final_df.columns = [c.lower() for c in final_df.columns] # FORCE LOWERCASE
    
    # 3. Save
    final_df.to_sql("play_by_play", ENGINE, if_exists='replace', index=False)
    
    # 4. Immediate Verification
    with ENGINE.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM play_by_play")).scalar()
        print(f"✅ VERIFIED: {result} rows are now sitting in Postgres!")

if __name__ == "__main__":
    get_data()