import json
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
FEATURE_COLUMNS = ["period", "seconds_remaining", "home_score", "away_score", "score_margin", "abs_score_margin", "total_score", "score_margin_per_minute", "is_home_leading", "is_tied", "is_late_game", "is_scoring_play"]

st.set_page_config(page_title="NBA CourtVision", page_icon="🏀", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
:root{--navy:#0b1220;--ink:#182030;--muted:#697386;--line:#e4e8ef;--canvas:#f6f7f9;--card:#fff;--orange:#f15a29;--soft:#fff1eb;--blue:#2f6fed}
html,body,[class*="css"]{font-family:"DM Sans",sans-serif}.stApp{background:var(--canvas);color:var(--ink)}.block-container{max-width:1380px;padding:2rem 2.4rem 4rem}h1,h2,h3,.brand-name,.page-title,.score-number,.metric-value{font-family:"Manrope",sans-serif;letter-spacing:-.035em}[data-testid="stHeader"]{background:transparent}
[data-testid="stSidebar"]{background:var(--navy);border:0}[data-testid="stSidebar"]>div{padding:1.25rem 1rem}[data-testid="stSidebar"] *{color:#dce3ef}[data-testid="stSidebar"] [data-testid="stRadio"] label{padding:.62rem .7rem;border-radius:10px;margin-bottom:.12rem}[data-testid="stSidebar"] [data-testid="stRadio"] label:hover{background:#ffffff12}[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked){background:#f15a2930}[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p{color:#fff;font-weight:700}[data-testid="stSidebar"] [data-testid="stRadio"]>label{display:none}[data-testid="stSidebar"] hr{border-color:#ffffff18}
.brand{display:flex;align-items:center;gap:.8rem;padding:.15rem .35rem 1.35rem}.brand-mark{width:42px;height:42px;border-radius:12px;background:var(--orange);color:white;display:grid;place-items:center;font-size:1.3rem;box-shadow:0 8px 24px #f15a2947}.brand-name{color:white;font-size:1.03rem;font-weight:800}.brand-sub{color:#8995a8;font-size:.72rem;margin-top:.12rem}.side-label{color:#718096;font-size:.68rem;font-weight:700;letter-spacing:.11em;text-transform:uppercase;padding:.2rem .45rem .55rem}.season-card{background:#ffffff0e;border:1px solid #ffffff14;border-radius:12px;padding:.85rem;margin-top:1.25rem}.season-label{color:#8290a5;font-size:.68rem;text-transform:uppercase;letter-spacing:.09em}.season-value{color:white;font-weight:700;margin-top:.28rem}.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:#36d399;margin-right:.4rem;box-shadow:0 0 0 4px #36d3991a}
.page-head{display:flex;justify-content:space-between;align-items:flex-end;gap:1.5rem;margin-bottom:1.6rem}.eyebrow{color:var(--orange);font-size:.72rem;font-weight:800;letter-spacing:.11em;text-transform:uppercase;margin-bottom:.38rem}.page-title{color:var(--ink);font-size:clamp(2rem,4vw,3.35rem);font-weight:800;line-height:1.05;margin:0}.page-copy{color:var(--muted);font-size:.98rem;max-width:680px;line-height:1.55;margin-top:.62rem}.data-badge{white-space:nowrap;background:#e9f8f2;color:#087c57;border:1px solid #c9ecdf;border-radius:999px;padding:.42rem .72rem;font-size:.76rem;font-weight:700}
.metric-card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:1.05rem 1.12rem;min-height:120px;box-shadow:0 5px 18px #141e3709}.metric-label{color:var(--muted);font-size:.7rem;font-weight:800;letter-spacing:.075em;text-transform:uppercase}.metric-value{color:var(--ink);font-size:1.72rem;font-weight:800;line-height:1.1;margin-top:.48rem}.metric-note{color:#8590a2;font-size:.75rem;line-height:1.35;margin-top:.43rem}.metric-accent{width:26px;height:3px;border-radius:2px;background:var(--orange);margin-bottom:.7rem}.section-head{display:flex;justify-content:space-between;align-items:center;margin:1.65rem 0 .75rem}.section-title{color:var(--ink);font-family:"Manrope",sans-serif;font-size:1.02rem;font-weight:800}.section-note{color:var(--muted);font-size:.76rem}.panel{background:white;border:1px solid var(--line);border-radius:14px;padding:1.1rem 1.2rem;box-shadow:0 5px 18px #141e3708}
.game-card{background:var(--navy);border-radius:18px;padding:1.45rem 1.6rem;color:white;position:relative;overflow:hidden}.game-card:after{content:"";position:absolute;width:240px;height:240px;border:40px solid #f15a291f;border-radius:50%;right:-100px;top:-115px}.game-meta{color:#91a0b6;font-size:.76rem;font-weight:700;text-transform:uppercase;letter-spacing:.075em}.score-grid{display:grid;grid-template-columns:1fr auto 1fr;gap:1.2rem;align-items:center;margin-top:1rem;position:relative;z-index:1}.team-away{text-align:right}.team-home{text-align:left}.team-name{color:#9aa7ba;font-size:.7rem;font-weight:700;text-transform:uppercase}.team-abbr{color:white;font-family:"Manrope";font-size:1.45rem;font-weight:800}.score-number{color:white;font-size:3.35rem;font-weight:800;line-height:1}.score-dash{color:#516078;padding:0 .3rem}.winner{color:#ffc0a9;font-size:.68rem;font-weight:800;margin-top:.3rem}.facts{display:flex;flex-wrap:wrap;gap:.48rem;margin-top:1.2rem;position:relative;z-index:1}.fact{background:#ffffff12;border:1px solid #ffffff14;border-radius:8px;padding:.4rem .62rem;color:#dbe3ef;font-size:.73rem}
.rank-row{display:grid;grid-template-columns:30px 1fr auto;gap:.7rem;align-items:center;border-bottom:1px solid var(--line);padding:.73rem 0}.rank-row:last-child{border:0}.rank-num{color:#a0a8b5;font-size:.75rem;font-weight:700}.rank-team{font-weight:800}.rank-record{color:var(--muted);font-size:.75rem}.rank-value{font-family:"Manrope";font-weight:800}.prob-card{background:var(--navy);color:white;border-radius:18px;padding:1.5rem;text-align:center}.prob-label{color:#9aa7ba;font-size:.72rem;font-weight:800;letter-spacing:.09em;text-transform:uppercase}.prob-value{font-family:"Manrope";font-size:4rem;font-weight:800;letter-spacing:-.07em;line-height:1;margin:.65rem 0 .25rem}.prob-team{color:#ffc0a9;font-weight:700;font-size:.85rem}.prob-track{height:9px;background:#293449;border-radius:10px;overflow:hidden;margin:1.25rem 0 .55rem}.prob-fill{height:100%;background:linear-gradient(90deg,#ff7a45,var(--orange));border-radius:10px}.prob-scale{display:flex;justify-content:space-between;color:#7e8ba1;font-size:.66rem}.insight{background:var(--soft);border:1px solid #ffd9ca;border-radius:12px;padding:.9rem 1rem;color:#8b3517;font-size:.84rem;line-height:1.5}.empty{background:white;border:1px dashed #cbd2dc;border-radius:14px;padding:2.2rem;text-align:center;color:var(--muted)}div[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:12px;overflow:hidden}[data-testid="stVegaLiteChart"]{background:white;border:1px solid var(--line);border-radius:14px;padding:.6rem}.stButton>button{border-radius:9px;font-weight:700}div[data-baseweb="select"]>div,div[data-baseweb="input"]>div{border-radius:9px}[data-testid="stExpander"]{background:white;border-color:var(--line);border-radius:12px}
@media(max-width:760px){.block-container{padding:1.4rem 1rem 3rem}.page-head{display:block}.data-badge{display:inline-block;margin-top:.8rem}.score-number{font-size:2.3rem}.score-grid{gap:.5rem}.game-card{padding:1.1rem}}
</style>""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_data():
    games = pd.read_csv(DATA_DIR / "games.csv")
    plays = pd.read_csv(DATA_DIR / "plays_with_predictions.csv")
    games["game_id"], plays["game_id"] = games["game_id"].astype(str), plays["game_id"].astype(str)
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")
    games["month"] = games["game_date"].dt.strftime("%B %Y")
    for col in ["sequence_number", "period", "seconds_remaining", "home_score", "away_score", "score_margin", "home_win_probability"]:
        plays[col] = pd.to_numeric(plays[col], errors="coerce")
    return games, plays

@st.cache_resource(show_spinner=False)
def load_model():
    model = XGBClassifier(); model.load_model(BASE_DIR / "models/nba_xgb_win_probability.json"); return model

@st.cache_data(show_spinner=False)
def game_summary(games, plays):
    rows = []
    for game_id, gp in plays.groupby("game_id", sort=False):
        scored = gp.dropna(subset=["home_score", "away_score"]).sort_values("sequence_number")
        predicted = gp.dropna(subset=["home_win_probability"]).sort_values("sequence_number")
        if scored.empty: continue
        final, margins = scored.iloc[-1], scored["score_margin"].dropna()
        signs = margins.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        hs, aws = int(final["home_score"]), int(final["away_score"])
        rows.append({"game_id":game_id,"home_score_final":hs,"away_score_final":aws,"total_score":hs+aws,"final_margin":hs-aws,"abs_final_margin":abs(hs-aws),
            "opening_home_win_probability":predicted["home_win_probability"].iloc[0] if len(predicted) else None,"final_home_win_probability":predicted["home_win_probability"].iloc[-1] if len(predicted) else None,
            "win_probability_swing":predicted["home_win_probability"].max()-predicted["home_win_probability"].min() if len(predicted) else None,"lead_changes":int(((signs.shift()*signs)<0).sum()),
            "largest_home_lead":margins.max() if len(margins) else None,"largest_away_lead":abs(margins.min()) if len(margins) else None,"total_plays":len(gp)})
    return games.merge(pd.DataFrame(rows), on="game_id", how="left")

@st.cache_data(show_spinner=False)
def team_summary(gs):
    rows=[]
    for team in sorted(set(gs.home_team_abbrev.dropna())|set(gs.away_team_abbrev.dropna())):
        home, away = gs[gs.home_team_abbrev==team], gs[gs.away_team_abbrev==team]
        played=pd.concat([home,away],ignore_index=True)
        wins=int((home.home_score_final>home.away_score_final).sum()+(away.away_score_final>away.home_score_final).sum())
        pf=pd.concat([home.home_score_final,away.away_score_final],ignore_index=True); pa=pd.concat([home.away_score_final,away.home_score_final],ignore_index=True)
        rows.append({"team":team,"games":len(played),"wins":wins,"losses":len(played)-wins,"win_pct":wins/len(played),"avg_points_for":pf.mean(),"avg_points_allowed":pa.mean(),"avg_margin":(pf-pa).mean(),"close_games":int((played.abs_final_margin<=5).sum())})
    return pd.DataFrame(rows).sort_values(["win_pct","avg_margin"],ascending=False).reset_index(drop=True)

def features(period, seconds, home, away, scoring):
    margin=home-away
    return pd.DataFrame([{"period":period,"seconds_remaining":seconds,"home_score":home,"away_score":away,"score_margin":margin,"abs_score_margin":abs(margin),"total_score":home+away,"score_margin_per_minute":margin/max(seconds/60,1/60),"is_home_leading":int(margin>0),"is_tied":int(margin==0),"is_late_game":int(seconds<=300),"is_scoring_play":int(scoring)}],columns=FEATURE_COLUMNS)

def pct(v,d=1): return "n/a" if pd.isna(v) else f"{v*100:.{d}f}%"
def num(v): return "n/a" if pd.isna(v) else f"{int(v):,}"
def header(kicker,title,copy): st.markdown(f'<div class="page-head"><div><div class="eyebrow">{escape(kicker)}</div><h1 class="page-title">{escape(title)}</h1><div class="page-copy">{escape(copy)}</div></div><div class="data-badge"><span class="dot"></span>2025–26 dataset loaded</div></div>',unsafe_allow_html=True)
def section(title,note=""): st.markdown(f'<div class="section-head"><div class="section-title">{escape(title)}</div><div class="section-note">{escape(note)}</div></div>',unsafe_allow_html=True)
def metric(label,value,note): st.markdown(f'<div class="metric-card"><div class="metric-accent"></div><div class="metric-label">{escape(str(label))}</div><div class="metric-value">{escape(str(value))}</div><div class="metric-note">{escape(str(note))}</div></div>',unsafe_allow_html=True)
def label(row): return f"{row.away_team_abbrev} at {row.home_team_abbrev} · {row.game_date.strftime('%b %-d, %Y')}"
def scorecard(g):
    aw,hw=g.away_score_final>g.home_score_final,g.home_score_final>g.away_score_final
    st.markdown(f'<div class="game-card"><div class="game-meta">{g.game_date.strftime("%B %-d, %Y")} · Final</div><div class="score-grid"><div class="team-away"><div class="team-name">Away</div><div class="team-abbr">{escape(g.away_team_abbrev)}</div>{"<div class=winner>WINNER</div>" if aw else ""}</div><div class="score-number">{num(g.away_score_final)}<span class="score-dash">–</span>{num(g.home_score_final)}</div><div class="team-home"><div class="team-name">Home</div><div class="team-abbr">{escape(g.home_team_abbrev)}</div>{"<div class=winner>WINNER</div>" if hw else ""}</div></div><div class="facts"><div class="fact">↕ {pct(g.win_probability_swing)} probability swing</div><div class="fact">⇄ {num(g.lead_changes)} lead changes</div><div class="fact">△ {num(g.abs_final_margin)}-point margin</div><div class="fact">▤ {num(g.total_plays)} plays</div></div></div>',unsafe_allow_html=True)
def rankings(df,limit=5):
    rows=''.join(f'<div class="rank-row"><div class="rank-num">{i+1:02}</div><div><div class="rank-team">{r.team}</div><div class="rank-record">{r.wins}–{r.losses} · {r.games} games</div></div><div class="rank-value">{pct(r.win_pct,0)}</div></div>' for i,r in df.head(limit).iterrows())
    st.markdown(f'<div class="panel">{rows}</div>',unsafe_allow_html=True)

games,plays=load_data(); gs=game_summary(games,plays); ts=team_summary(gs); teams=sorted(ts.team.tolist())
metrics=json.loads((BASE_DIR/"artifacts/training_metrics.json").read_text()); model=load_model()
pages={"Overview":"⌂","Game Explorer":"↗","Teams":"◎","Prediction Lab":"✦","Model":"◇"}
with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-mark">◒</div><div><div class="brand-name">CourtVision</div><div class="brand-sub">NBA AI ANALYTICS</div></div></div><div class="side-label">Workspace</div>',unsafe_allow_html=True)
    page=st.radio("Navigate",list(pages),format_func=lambda x:f"{pages[x]}   {x}",label_visibility="collapsed")
    st.markdown('---');st.markdown('<div class="season-card"><div class="season-label">Active season</div><div class="season-value"><span class="dot"></span>2025–26 NBA</div></div>',unsafe_allow_html=True)
    st.caption("Analytical estimates, not betting advice.")

if page=="Overview":
    header("Season command center","See the season at a glance.","Start with the big picture, spot the strongest teams, and jump into the latest completed games.")
    items=[("Games analyzed",num(len(gs)),"Full 2025–26 schedule"),("Model-scored plays",num(plays.home_win_probability.notna().sum()),"Play-by-play estimates"),("Model ROC AUC",f"{metrics['roc_auc']:.3f}","Held-out evaluation"),("Average game swing",pct(gs.win_probability_swing.mean()),"Low-to-high probability")]
    for c,item in zip(st.columns(4),items):
        with c: metric(*item)
    left,right=st.columns([1.65,1],gap="large")
    with left:
        section("Latest final","Most recent game in the dataset");scorecard(gs.sort_values("game_date",ascending=False).iloc[0]);section("Recent games","Latest 8 results")
        recent=gs.sort_values("game_date",ascending=False).head(8).copy();recent["Result"]=recent.apply(lambda r:f"{r.away_team_abbrev} {num(r.away_score_final)} – {num(r.home_score_final)} {r.home_team_abbrev}",axis=1)
        st.dataframe(recent[["game_date","Result","abs_final_margin","lead_changes"]],width="stretch",hide_index=True,height=320,column_config={"game_date":st.column_config.DatetimeColumn("Date",format="MMM D"),"abs_final_margin":"Margin","lead_changes":"Lead changes"})
    with right:
        section("Top teams","Ranked by win percentage");rankings(ts);section("Season pulse");close=int((gs.abs_final_margin<=5).sum());st.markdown(f'<div class="insight"><b>{close:,} games</b> finished within five points — {pct(close/len(gs),0)} of the dataset.</div>',unsafe_allow_html=True)

elif page=="Game Explorer":
    header("Game explorer","Find any game. Read every shift.","Filter the season, select a matchup, and move from the final score to the play-level story.")
    with st.expander("Filter games",expanded=True):
        a,b,c=st.columns(3);tf=a.selectbox("Team",["All teams"]+teams);mf=b.selectbox("Month",["All months"]+list(games.month.dropna().unique()));sf=c.slider("Minimum probability swing",0,100,0,10,format="%d%%")
    filtered=gs.copy()
    if tf!="All teams": filtered=filtered[(filtered.home_team_abbrev==tf)|(filtered.away_team_abbrev==tf)]
    if mf!="All months": filtered=filtered[filtered.month==mf]
    filtered=filtered[filtered.win_probability_swing.fillna(0)>=sf/100].sort_values("game_date",ascending=False)
    if filtered.empty: st.markdown('<div class="empty"><b>No games match these filters.</b><br>Try widening your search.</div>',unsafe_allow_html=True)
    else:
        gid=st.selectbox(f"Choose a game · {len(filtered):,} results",filtered.game_id,format_func=lambda x:label(filtered[filtered.game_id==x].iloc[0]));g=gs[gs.game_id==gid].iloc[0];gp=plays[plays.game_id==gid].sort_values("sequence_number");pred=gp.dropna(subset=["home_win_probability"]);scorecard(g)
        chart,stats=st.columns([1.7,1],gap="large")
        with chart:
            section("Win probability",f"{g.home_team_abbrev} probability by play");data=pred[["sequence_number","home_win_probability"]].rename(columns={"sequence_number":"Play","home_win_probability":"Home win probability"});st.area_chart(data,x="Play",y="Home win probability",height=360,color="#f15a29")
        with stats:
            section("Game snapshot");x,y=st.columns(2)
            with x:metric("Largest home lead",num(g.largest_home_lead),g.home_team_abbrev)
            with y:metric("Largest away lead",num(g.largest_away_lead),g.away_team_abbrev)
            x,y=st.columns(2)
            with x:metric("Opening home WP",pct(g.opening_home_win_probability),g.home_team_abbrev)
            with y:metric("Final home WP",pct(g.final_home_win_probability),g.home_team_abbrev)
        section("Score margin","Above zero favors the home team");margin=gp[["sequence_number","score_margin"]].dropna().rename(columns={"sequence_number":"Play","score_margin":"Home score margin"});st.line_chart(margin,x="Play",y="Home score margin",height=240,color="#2f6fed")
        section("Play-by-play","Search and narrow the game log");x,y=st.columns([2,1]);query=x.text_input("Search plays",placeholder="Player, shot, foul…");periods=sorted(int(q) for q in gp.period.dropna().unique());q=y.selectbox("Period",["All periods"]+periods);log=gp.copy()
        if query:log=log[log.text.fillna("").str.contains(query,case=False,regex=False)]
        if q!="All periods":log=log[log.period==q]
        st.caption(f"Showing {len(log):,} of {len(gp):,} plays");st.dataframe(log[["period","clock_display","away_score","home_score","home_win_probability","play_type_text","text"]],width="stretch",height=520,hide_index=True,column_config={"period":"Q","clock_display":"Clock","away_score":g.away_team_abbrev,"home_score":g.home_team_abbrev,"home_win_probability":st.column_config.ProgressColumn(f"{g.home_team_abbrev} WP",min_value=0,max_value=1,format="%.3f"),"play_type_text":"Type","text":st.column_config.TextColumn("Play",width="large")})

elif page=="Teams":
    header("Team performance","Compare the league in seconds.","Open any team for its record, scoring profile, recent results, and league position.")
    selected=st.selectbox("Team profile",teams,index=teams.index("BOS") if "BOS" in teams else 0);t=ts[ts.team==selected].iloc[0];rank=int(ts.index[ts.team==selected][0])+1
    items=[("League rank",f"#{rank}","By win percentage"),("Record",f"{t.wins}–{t.losses}",f"{t.games} games"),("Win rate",pct(t.win_pct,0),"Completed games"),("Points / game",f"{t.avg_points_for:.1f}","Scoring average"),("Avg margin",f"{t.avg_margin:+.1f}","Points per game")]
    for c,item in zip(st.columns(5),items):
        with c:metric(*item)
    left,right=st.columns([1.3,1],gap="large")
    with left:
        section(f"{selected} recent results","Latest 12 games");tg=gs[(gs.home_team_abbrev==selected)|(gs.away_team_abbrev==selected)].sort_values("game_date",ascending=False).head(12)
        rows=[]
        for _,r in tg.iterrows():
            home=r.home_team_abbrev==selected;sc=r.home_score_final if home else r.away_score_final;allow=r.away_score_final if home else r.home_score_final;opp=r.away_team_abbrev if home else r.home_team_abbrev;rows.append({"Date":r.game_date,"W/L":"W" if sc>allow else "L","Opponent":opp,"Score":f"{int(sc)}–{int(allow)}","Margin":int(sc-allow)})
        st.dataframe(pd.DataFrame(rows),width="stretch",hide_index=True,height=455,column_config={"Date":st.column_config.DatetimeColumn(format="MMM D, YYYY")})
    with right:section("League table","Top 10 by win percentage");rankings(ts,10)
    section("Full league comparison","Click a header to sort");st.dataframe(ts,width="stretch",height=620,hide_index=True,column_config={"team":"Team","games":"GP","wins":"W","losses":"L","win_pct":st.column_config.ProgressColumn("Win %",min_value=0,max_value=1,format="%.3f"),"avg_points_for":st.column_config.NumberColumn("PF/G",format="%.1f"),"avg_points_allowed":st.column_config.NumberColumn("PA/G",format="%.1f"),"avg_margin":st.column_config.NumberColumn("Margin",format="%+.1f"),"close_games":"Close games"})

elif page=="Prediction Lab":
    header("Prediction lab","Turn a game state into a probability.","Set the quarter, clock, and score. The model estimates the home team's chance of winning from that moment.")
    inputs,output=st.columns([1.25,1],gap="large")
    with inputs:
        section("1. Describe the game")
        with st.container(border=True):
            x,y=st.columns(2);away=(x.text_input("Away team","AWAY",max_chars=12) or "AWAY").upper();home=(y.text_input("Home team","HOME",max_chars=12) or "HOME").upper();x,y=st.columns(2);period=x.selectbox("Period",range(1,9),index=3,format_func=lambda q:f"Quarter {q}" if q<=4 else f"Overtime {q-4}");maxmin=12 if period<=4 else 5;minutes=y.number_input("Minutes left in period",0,maxmin,min(5,maxmin));x,y=st.columns(2);aws=x.number_input(f"{away} score",0,250,95);hs=y.number_input(f"{home} score",0,250,98);scoring=st.checkbox("This follows a scoring play");seconds=int((4-period)*720+minutes*60) if period<=4 else int(minutes*60);st.caption(f"Model input: {num(seconds)} seconds remaining")
    f=features(period,seconds,hs,aws,scoring);p=float(model.predict_proba(f)[0][1]);fav=home if p>=.5 else away;favp=p if p>=.5 else 1-p
    with output:
        section("2. Read the estimate");st.markdown(f'<div class="prob-card"><div class="prob-label">Home win probability</div><div class="prob-value">{pct(p,0)}</div><div class="prob-team">{escape(fav)} is favored at {pct(favp,0)}</div><div class="prob-track"><div class="prob-fill" style="width:{p*100:.1f}%"></div></div><div class="prob-scale"><span>{escape(away)} 100%</span><span>EVEN</span><span>{escape(home)} 100%</span></div></div>',unsafe_allow_html=True);margin=hs-aws;state="tied" if margin==0 else f"{home if margin>0 else away} leads by {abs(margin)}";st.markdown(f'<div class="insight" style="margin-top:1rem"><b>Game state:</b> {escape(state)} with {minutes} minute(s) left. Adjust any input to recalculate instantly.</div>',unsafe_allow_html=True)
    with st.expander("See the model inputs"):st.dataframe(f,width="stretch",hide_index=True)

else:
    header("Model transparency","Know what powers the probability.","Review evaluation results, training scale, and the signals used before trusting an estimate.")
    items=[("ROC AUC",f"{metrics['roc_auc']:.3f}","Ranking performance"),("Calibration",pct(metrics["calibration_accuracy"]),"Probability reliability"),("Log loss",f"{metrics['log_loss']:.3f}","Lower is better"),("Brier score",f"{metrics['brier_score']:.3f}","Lower is better")]
    for c,item in zip(st.columns(4),items):
        with c:metric(*item)
    left,right=st.columns([1.2,1],gap="large")
    with left:
        section("How it works");st.markdown('<div class="panel"><div class="rank-row"><div class="rank-num">01</div><div><div class="rank-team">Read the game state</div><div class="rank-record">Quarter, clock, score, margin, and play context</div></div><div class="rank-value">INPUT</div></div><div class="rank-row"><div class="rank-num">02</div><div><div class="rank-team">Transform the signals</div><div class="rank-record">Create lead, tie, late-game, and pace features</div></div><div class="rank-value">FEATURES</div></div><div class="rank-row"><div class="rank-num">03</div><div><div class="rank-team">Estimate the outcome</div><div class="rank-record">XGBoost returns a home win probability</div></div><div class="rank-value">MODEL</div></div></div>',unsafe_allow_html=True);section("Important context");st.markdown('<div class="insight">This portfolio model does not account for injuries, lineups, travel, or betting-market information and should not be used as financial advice.</div>',unsafe_allow_html=True)
    with right:
        section("Training snapshot");x,y=st.columns(2)
        with x:metric("Training games",num(metrics["train_games"]),num(metrics["train_rows"])+" play states")
        with y:metric("Test games",num(metrics["test_games"]),num(metrics["test_rows"])+" play states")
        section("Feature set",f"{len(FEATURE_COLUMNS)} model inputs");st.markdown('<div class="panel">'+''.join(f'<span style="display:inline-block;background:#f2f4f7;border-radius:7px;padding:.35rem .5rem;margin:.2rem;font-size:.73rem;color:#596477">{escape(x.replace("_"," ").title())}</span>' for x in FEATURE_COLUMNS)+'</div>',unsafe_allow_html=True)
