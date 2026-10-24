#!/usr/bin/env bash
# Download the competition files into data/ (gitignored).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
kaggle competitions download arc-prize-2026-arc-agi-2 -p data
(cd data && unzip -oq ./*.zip && rm -f ./*.zip)
ls -la data
