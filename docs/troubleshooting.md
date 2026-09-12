# Troubleshooting

**`... not found. Run neurofly download first.`**
The connectome files are missing from `data/malecns/` (or from `NEUROFLY_DATA`). Run
`neurofly download`.

**`No module named 'neurofly'`**
The package is not installed in the active environment. `pip install -e .` from the
project root, in the environment you run the scripts with.

**The brain is silent (0 spikes) or saturated (most neurons firing).**
Drive is too weak or too strong for the subset. Raise or lower `--brain-gain` first (it
multiplies every synapse), then the encoder gain of the input in use (`--encoder-gain`,
`--retina-gain`, `--audio-gain`). `bench` prints how many neurons are active under a
constant drive; `watch --policy zero` and `play --dry-run` print spike counts.

**`retina mode 'hex' needs columnar optic-lobe neurons with hex coordinates`**
The `central` subset has no optic lobe. Use `--subset visual` or `brain`, or let
`--retina-mode auto` fall back to the projection mode.

**`no auditory neurons in this connectome subset`**
Sound needs the head: `central`, `visual` or `brain`, not `vnc`.

**`task 'pc' needs --brain malecns (or synthetic)`**
PC tasks always run through the brain; the controls are read out of neurons.

**`no visible window with '...' in its title`**
The title match is case-insensitive and partial, but the window must be visible and not
minimised. On Linux and macOS window lookup is unavailable; pass `--region`.

**`empty capture region ...; is the window minimised?`**
The window's client area has zero size. Restore it.

**`No input device matching '...'`**
`--audio` names a capture device that does not exist. `neurofly devices` lists
them; `loopback` captures what the PC plays.

**`no loopback capture for output '...'`**
The default output device has no loopback endpoint. Change the default output in the
system's sound settings, or capture a device instead.

**Keys are sent but the program ignores them.**
Keys go through the system's input events. Programs that read the keyboard at a lower level
or reject synthetic input will not see them; nothing in this project can change that.

**The mouse moves but the recording shows no motion.**
Recordings measure the cursor's position; a program that hides and re-centres the cursor
defeats that. Keyboard-only layouts still record fine.

**`imitate` scores every control near its base rate.**
The readout does not carry the information. Try `--subset visual` (the real retina),
`--brain-ms 20`, a wider readout (`--readout descending+cbmotor+visual`), or the raw
features (`--include-frame`, `--include-audio`) to confirm the recording itself is
learnable.

**`decoder was trained for controls [...], this layout has [...]`**
A decoder only fits the layout it was trained with. Pass the same `--keys`, `--buttons`,
`--mouse`, `--scroll`, or let `--run` supply them.

**`the live screen is one environment: using --n-envs 1`**
Expected: live training cannot run several copies of the screen. Train offline on
recordings or videos to use more processes.

**Windows: training hangs at start with many processes.**
Each process loads the connectome; the first load builds the cache and takes a minute. Run
`inspect` once first so the cache exists, then train.

**`the event backend needs numba and a CPU device`**
`--backend event` was forced on CUDA or without numba. Use `auto` (the default) or `torch`.

**The video from `watch` has an odd size or fails to encode.**
mp4 needs even dimensions; the writer is told to accept any size, but very unusual frames
(single-channel, alpha) are converted first. Recordings store 320 x 240 by default.

**Esc did not stop `play`.**
The panic key is read by a background listener; another program capturing the keyboard
exclusively can hide it. Ctrl+C in the terminal also stops the script, and every exit path
releases all keys and buttons.
