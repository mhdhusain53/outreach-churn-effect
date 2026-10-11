#!/bin/bash
set -euo pipefail

python3 /solution/solve.py
cp /solution/solve.py /app/output/analyze.py
cp /solution/surv_lib.py /app/output/surv_lib.py
