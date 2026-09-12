# Run the test suite. Extra arguments go to pytest:
#   scripts\test.ps1                       everything (about three minutes)
#   scripts\test.ps1 -k body               a selection
#   scripts\test.ps1 --cov                 with the coverage report
#   scripts\test.ps1 tests\core            one directory
# Uses the Python on PATH, or the one in NEUROFLY_PYTHON.
$python = if ($env:NEUROFLY_PYTHON) { $env:NEUROFLY_PYTHON } else { "python" }
Set-Location (Split-Path -Parent $PSScriptRoot)
& $python -m pytest -q --no-header -p no:cacheprovider @args
exit $LASTEXITCODE
