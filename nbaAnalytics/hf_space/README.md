---
title: NBA AI Analytics Dashboard
emoji: 🏀
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# NBA AI Analytics Dashboard

Interactive demo for an NBA play-by-play analytics pipeline:

- ESPN play-by-play warehouse sample
- XGBoost win-probability model
- Game-level win-probability playback
- Ad hoc inference controls
- Model evaluation metrics

This Space uses a bundled demo export from the local PostgreSQL warehouse. The full project includes the ETL, training, scoring, FastAPI serving, Docker Compose, and deployment scripts.
