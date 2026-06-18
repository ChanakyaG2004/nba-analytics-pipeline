import argparse
import json
import statistics
import time

import requests


DEFAULT_PAYLOAD = {
    "period": 4,
    "seconds_remaining": 300,
    "home_score": 98,
    "away_score": 95,
    "scoring_play": False,
}


def percentile(values, percentile_value):
    if not values:
        return 0

    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile_value)
    return ordered[index]


def run_benchmark(args):
    url = args.url.rstrip("/") + "/predict"
    latencies = []
    failures = 0

    for _ in range(args.warmup):
        requests.post(url, json=DEFAULT_PAYLOAD, timeout=args.timeout).raise_for_status()

    started_at = time.perf_counter()

    for _ in range(args.requests):
        request_start = time.perf_counter()
        try:
            response = requests.post(url, json=DEFAULT_PAYLOAD, timeout=args.timeout)
            response.raise_for_status()
        except requests.RequestException:
            failures += 1
            continue

        latencies.append((time.perf_counter() - request_start) * 1000)

    total_seconds = time.perf_counter() - started_at
    successful = len(latencies)

    return {
        "url": url,
        "requests": args.requests,
        "successful": successful,
        "failures": failures,
        "throughput_rps": successful / total_seconds if total_seconds else 0,
        "latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else 0,
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "max": max(latencies) if latencies else 0,
        },
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Benchmark the NBA win-probability API.")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=5)
    return parser


if __name__ == "__main__":
    print(json.dumps(run_benchmark(build_parser().parse_args()), indent=2))
