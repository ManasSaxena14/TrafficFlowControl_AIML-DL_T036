#!/usr/bin/env bash
# Download METR-LA (+ PEMS-BAY for the Phase 9 extension) and the sensor graph into data/raw/.
# Source: https://github.com/liyaguang/DCRNN#data-preparation
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw ref
[ -d ref/DCRNN ] || git clone --depth 1 https://github.com/liyaguang/DCRNN.git ref/DCRNN
cp ref/DCRNN/data/sensor_graph/* data/raw/
if [ ! -f data/raw/metr-la.h5 ]; then
  tmp=$(mktemp -d)
  gdown --folder https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX -O "$tmp"
  find "$tmp" -name "*.h5" -exec mv {} data/raw/ \;
  rm -rf "$tmp"
fi
ls -lh data/raw
