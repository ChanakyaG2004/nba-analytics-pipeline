# Hugging Face Spaces Deployment

This deployment is free-friendly and does not require AWS.

The Space is self-contained:

- Streamlit app
- compact exported demo dataset
- trained XGBoost model artifact
- model metrics
- Dockerfile exposing port `7860`

## Login

Create a Hugging Face access token with write access:

https://huggingface.co/settings/tokens

Then log in from your terminal:

```bash
huggingface-cli login
```

Paste the token when prompted. Do not paste tokens into chat.

## Local Test

```bash
cd nbaAnalytics/hf_space
streamlit run app.py --server.port 7860
```

Open:

```text
http://localhost:7860
```

## Deploy

```bash
cd nbaAnalytics
python3 deploy/deploy_hf_space.py
```

Or choose a specific Space repo name:

```bash
python3 deploy/deploy_hf_space.py --repo-id your-username/nba-ai-analytics-dashboard
```

The script prints the public Space URL after upload.

## Refresh Demo Data

After changing the local warehouse/model, refresh the bundled data by rerunning the export step from the project root:

```bash
python3 src/score_predictions.py
```

Then regenerate `hf_space/data/*.csv` from the local Postgres warehouse before redeploying.
