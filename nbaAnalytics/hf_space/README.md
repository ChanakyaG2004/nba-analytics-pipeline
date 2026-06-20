---
title: NBA AI Analytics Dashboard
emoji: 🏀
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# NBA AI Analytics Dashboard

Interactive 2025-26 NBA analytics dashboard built from a local PostgreSQL warehouse export:

- 1,401 loaded 2025-26 season games
- 679,005 play-by-play rows
- 601,601 scored play-level win probabilities
- XGBoost win-probability model
- Game center with win-probability playback
- Season board and team-stat views
- Ad hoc inference controls
- Model evaluation metrics

This Space uses bundled CSV data and a saved model artifact so it can run publicly without a live database. The full project includes the ETL, PostgreSQL warehouse, training, scoring, FastAPI serving, Docker Compose, and deployment scripts.
