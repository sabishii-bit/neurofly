# Examples

Starting points to copy into your own project. Nothing here is imported by the package.

| File | What it shows |
|---|---|
| `health_bar_task.py` | a `Task` that reads a bar on the screen for reward, ends the episode when it empties, and presses a key to restart |

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
