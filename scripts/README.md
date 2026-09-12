# Scripts

| Script | What it does |
|---|---|
| `test.ps1` / `test.sh` | run the test suite; arguments go to pytest (`-k body`, `--cov`, a path) |
| `demo.py body` | the fly's limbs cycling a tripod gait in the Three.js viewer (`--walk physics` runs it through the dynamics instead; `--walk real` plays a real fly's walking from flybody's dataset, downloaded on first use) |
| `demo.py brain` | the brain atlas lit by a run's activity |
| `demo.py workbench` | what the fly saw, the brain and the body on one timeline |
| `demo.py web-fps` | the Three.js game training a served brain in the page |
| `demo.py all` | build every demo asset without opening a browser |

The demos build their assets into `assets/demo/` on first use (the body ones need the
`flybody` extra; the brain ones use the downloaded connectome when it is there and the
small `toy` brain otherwise) and start a local static server, then open the page.
`--rebuild` makes the assets again, `--no-open` only prints the URL. Run them with the
Python that has neurofly installed: `python scripts/demo.py body`.
