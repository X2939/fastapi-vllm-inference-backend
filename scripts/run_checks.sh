#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

python3 -m compileall -q \
    app attention benchmarks engine experiments scripts tests visualization
python3 -m pytest
