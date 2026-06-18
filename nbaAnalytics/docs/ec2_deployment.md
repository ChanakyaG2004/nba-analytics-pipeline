# EC2 Deployment Runbook

This runbook deploys the trained FastAPI inference service and Streamlit dashboard to a single AWS EC2 host with Docker Compose.

Creating an EC2 instance can incur AWS charges. Terminate the instance when the demo is done.

## Automated Deploy

Configure AWS credentials first:

```bash
aws configure
```

Then run:

```bash
cd nbaAnalytics
AWS_REGION=us-west-2 INSTANCE_TYPE=t4g.small ./deploy/deploy_ec2.sh
```

The script will:

- create or reuse an EC2 key pair
- create or reuse a security group
- allow SSH, API, and dashboard access from your current public IP
- launch Ubuntu 22.04 ARM64
- install Docker on the instance
- copy this project to EC2
- start `db`, `api`, and `dashboard` with Docker Compose
- print the public API/dashboard URLs

The default SSH key is written to:

```bash
~/.ssh/nba-analytics-key.pem
```

## Instance

Recommended baseline:

- Ubuntu 22.04 LTS
- `t3.small` or larger
- Security group inbound rules:
  - SSH: `22` from your IP
  - API: `8000` from your IP or trusted range
  - Dashboard: `8501` from your IP or trusted range

## Host Setup

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl git
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker ubuntu
```

Log out and back in after adding the user to the `docker` group.

## Deploy

```bash
git clone <your-repo-url>
cd nba-analytics-pipeline/nbaAnalytics
docker compose up -d --build db api dashboard
docker compose ps
```

Check API health:

```bash
curl -s http://localhost:8000/health
```

Call prediction:

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"period":4,"seconds_remaining":300,"home_score":98,"away_score":95}'
```

## Benchmark

Run from the EC2 host:

```bash
python3 -m pip install requests
python3 src/benchmark_api.py --url http://127.0.0.1:8000 --requests 1000 --warmup 50
```

Run from your laptop against the EC2 public IP:

```bash
python3 src/benchmark_api.py --url http://<ec2-public-ip>:8000 --requests 1000 --warmup 50
```

Use the remote benchmark number for any EC2 latency claim. Local laptop latency and EC2 latency are different measurements.

## Operations

View logs:

```bash
docker compose logs -f api
```

Restart API after model changes:

```bash
docker compose up -d --build api
```

Stop services:

```bash
docker compose down
```
