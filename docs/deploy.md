# Hosting the brain

The runtime does not have to run on the machine that uses it. `neurofly-core serve --ws`
is an ordinary WebSocket service, so a brain can live in a container on a server while a
web app, a game or a robot talks to it from anywhere. This page is the deployment shape;
the client side is in [From your own project](from-your-project.md).

## What stays local, and what does not have to

* **Nothing needs your PC** except software that only exists there. A game that renders
  in a browser, a Node process, a Unity build with a WebSocket client: all of these can
  talk to a hosted brain. The `record`, `train` and `play` commands that capture *your*
  screen and press *your* keys are the local case, for software that cannot run
  headlessly elsewhere; they are an option, not a requirement.
* **Training happens where the frames are made.** Policy training over the protocol
  (`observe`, `set_policy`, `save`) and in-brain plasticity (`reward` on the step) work
  against a hosted brain exactly as against a local one. The Python trainers that need a
  recording (`imitate`, `surrogate`, `detect-label`) work on recordings a hosted brain
  writes for you (`record` op), in a training container next to it.

## The runtime container

```powershell
docker build -f docker/Dockerfile.core -t neurofly-core .
docker run -p 8765:8765 -e NEUROFLY_TOKEN=secret \
    -v ./artifacts/myapp:/artifact:ro -v ./data/recordings:/recordings neurofly-core
```

That serves the artifact at `ws://host:8765` with:

* **A brain per client** (`--per-client`). Every connection gets its own fresh model, so
  two players or two trainers never share a state. Without it (the default of a bare
  `serve`) all connections share one brain, which is what a viewer watching a driver
  wants.
* **A token** (`--token`, or the `NEUROFLY_TOKEN` variable). Clients pass it as
  `?token=...` on the URL (what the bindings do) or as a first message
  `{"op": "hello", "token": ...}`. Without the right token the connection is refused.
* **Allowed origins** (`--origins https://mygame.example`) if you want browsers from
  anywhere else refused. Unset means any origin.
* **Recordings** go to `/recordings` (the path the client names, relative to the
  container's working directory or absolute); mount it so the training container sees it.

Put a TLS-terminating reverse proxy in front for anything public (Caddy does it in two
lines; nginx, Traefik, a cloud load balancer likewise). Behind it the service is plain
`ws://`; the browser sees `wss://`. The container has no TLS of its own on purpose.

`docker/compose.yaml` runs the brain and, on demand, the training image.

## The training container

```powershell
docker compose -f docker/compose.yaml run --rm training \
    neurofly download                                                     # the data, once
docker compose -f docker/compose.yaml run --rm training \
    neurofly build /artifacts/base --brain malecns --keys w,a,d --detect "owl2:enemy"
docker compose -f docker/compose.yaml run --rm training \
    neurofly imitate /data/recordings/web_171234 --brain malecns --run-name web1
docker compose -f docker/compose.yaml run --rm training \
    neurofly export /runs/web1 /artifacts/myapp-v2
```

`Dockerfile.training` has the training package with the detector extras; add
`training[flybody]` for the body tasks. The loop for a web game is: the game records a
session on the hosted brain, `imitate` or `surrogate` trains on it in the training
container, `export` writes the artifact, and the runtime container is restarted on it
(or a second one is started next to it and the client pointed at it).

## Capacity

A central-brain model is about 50 MB of arrays and steps a 10 ms observation in one to
three milliseconds on one CPU core; `--per-client` loads a fresh copy per connection (a
second or two). One core per active client is a fair rule of thumb for 10 to 30 frames a
second; the frame itself is a 20 KB JPEG at 320 by 240. A GPU is not needed for serving.
Frames are decoded from PNG or JPEG in the runtime, so send those rather than raw RGB over
a network.

## Local, when the software is local

The same client talks to `neurofly-core serve artifacts/x --ws 127.0.0.1:8765` on your
own PC, and `neurofly play` / `neurofly-core run` drive local software directly. A
recording made locally (`neurofly record`) and one made by a hosted brain are the same
format, so a project can start local and move to a server without changing anything but
the URL.
