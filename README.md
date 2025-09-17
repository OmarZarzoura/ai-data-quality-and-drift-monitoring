# AI Data Quality & Drift Monitoring Platform

Detect **schema changes**, **data drift** (KS-test, PSI), and quality regressions across ML pipelines.

## Features
- `POST /baseline` to store reference dataset
- `POST /monitor` to compare current batch vs baseline per-column
- KS-test p-value & PSI thresholds out of the box
- Dockerized API, CI, tests

## Run
```bash
docker build -t dq-drift .
docker run -p 9000:9000 dq-drift
```
