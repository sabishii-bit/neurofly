# Examples

Starting points to copy into your own project. Nothing here is imported by the package.

| File | What it shows |
|---|---|
| `health_bar_task.py` | a `Task` that reads a bar on the screen for reward, ends the episode when it empties, and presses a key to restart |
| `workbench.html` | the three-panel workbench: what the fly saw, the brain atlas (committed under `assets/brain-atlas`, loaded by default) lit by its activity, the body, on one timeline with play, scrub, speed and loop; takes neurofly's own files or any replay-format model output, by URL or dropped onto the brain panel, validated in the page |
| `web_fps.html` | a Three.js game that trains the fly in the browser over a WebSocket (evolution strategies in the page), records the session for the Python trainers, and saves the artifact; the template for a web game against a local or hosted brain |
| `brain_viewer.html` | the brain as a point cloud lit by its spikes, from `--activity-out` files or an `--activity-ws` / `serve --ws` stream; synapses from an artifact |
| `three_viewer.html` | the exported fly body animated from `watch --poses` files, a `--pose-ws` stream, or `neurofly body-serve` (then with buttons that drive the fly) |
| `node_consumer.js`, `node_train_es.js` | driving and training a served brain from Node |

Run one against a window, without touching anything, to check that the reward reads what
you think it reads:

```powershell
neurofly watch --task pc --brain malecns --window "My App" --keys w,a,s,d \
    --reward examples/health_bar_task.py:HealthBar --max-steps 50
```

Then train with it:

```powershell
neurofly train --task pc --brain malecns --window "My App" --keys w,a,s,d --mouse \
    --reward examples/health_bar_task.py:HealthBar --max-steps 600
```

See `docs/pc.md` and `docs/extending.md` for the interfaces.
