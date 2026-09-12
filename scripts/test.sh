#!/usr/bin/env sh
# Run the test suite. Extra arguments go to pytest:
#   scripts/test.sh                        everything (about three minutes)
#   scripts/test.sh -k body                a selection
#   scripts/test.sh --cov                  with the coverage report
# Uses the Python on PATH, or the one in NEUROFLY_PYTHON.
cd "$(dirname "$0")/.." || exit 1
exec "${NEUROFLY_PYTHON:-python}" -m pytest -q --no-header -p no:cacheprovider "$@"
