#!/usr/bin/env bash
set -e
export PATH="/home/eudespatricio/.local/bin:$PATH"
if [ $# -eq 0 ]; then
    pytest -v /home/eudespatricio/ia_trader/tests/test_*.py
else
    pytest -v "$@"
fi
