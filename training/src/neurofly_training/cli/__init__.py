"""The neurofly command: one entry point for building, training and exporting.

    neurofly list                                    # body tasks, PC tasks, subsets
    neurofly devices                                 # audio capture devices
    neurofly download                                # fetch the connectome files
    neurofly train   --task forward --brain malecns  # PPO (body or PC)
    neurofly train   --task pc --brain malecns --window "My App" --keys w,a,s,d --mouse \
                     --reward my_project.py:MyTask
    neurofly es      --task pc --brain malecns --window "My App" --keys w,a,s,d \
                     --reward my_project.py:MyTask
    neurofly record  --window "My App" --keys w,a,s,d --mouse --audio loopback \
                     --out data/recordings/run1
    neurofly imitate data/recordings/run1 --run-name imitate_myapp
    neurofly play    --window "My App" --run runs/imitate_myapp
    neurofly watch   runs/<run> --video videos/<run>.mp4
    neurofly export  runs/imitate_myapp artifacts/myapp   # -> neurofly-core, any language
    neurofly build   artifacts/base --brain malecns --keys w,a,s,d --mouse   # train it elsewhere
    neurofly eval    artifacts/myapp data/recordings/run1 --reward my_project.py:MyTask
    neurofly surrogate data/recordings/run1 --brain malecns --epochs 5   # train through the brain
    neurofly detect-label footage/*.mp4 --detect "owl2:enemy,health pack" --out data/objects
    neurofly detect-train data/objects --out runs/det1   # then --detect runs/det1 anywhere
    neurofly bench | inspect                          # brain tools

Every subcommand takes --help. ``python -m neurofly_training`` is the same thing.
"""
from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "train": ("train", "PPO on top of the brain readout (body tasks and the PC)"),
    "es": ("train_es", "evolution strategies over the neuron-to-output map"),
    "imitate": ("train_imitation", "fit the neuron-to-control map to a recording of you"),
    "record": ("record_pc", "record your own use of the PC: frames, sound, inputs"),
    "play": ("play_pc", "let the brain use the PC: keyboard and mouse out"),
    "watch": ("watch", "roll out a run and write a video"),
    "export": ("export", "a run directory -> an artifact for neurofly-core"),
    "build": ("build", "a base artifact (brain + encoders, no policy) to train from any language"),
    "eval": ("eval", "score an artifact offline against recordings"),
    "calibrate": ("calibrate", "sweep a gain for sparse, not silent, brain activity"),
    "idm": ("idm", "train an inverse dynamics model on labelled recordings"),
    "label": ("label", "label footage that has no input log, with an inverse dynamics model"),
    "detect-label": ("detect_label", "auto-label objects in footage with a detector (or "
                                     "preview one), as a YOLO-layout dataset"),
    "detect-train": ("detect_train", "fine-tune a fast object detector on such a dataset"),
    "replay": ("replay", "a video with the brain's spikes and controls drawn beside the frames"),
    "export-body": ("export_body", "the fly body as a glTF, for Three.js and other renderers"),
    "surrogate": ("surrogate", "train through the brain with surrogate gradients"),
    "bench": ("bench_brain", "how fast the brain steps on this machine"),
    "inspect": ("inspect_connectome", "what is in the connectome"),
    "download": ("download_data", "fetch the connectome files"),
}


def usage() -> str:
    lines = [__doc__.strip(), "", "commands:"]
    lines.append("  list      body tasks, PC tasks and connectome subsets")
    lines.append("  devices   audio capture devices")
    for name, (_, desc) in COMMANDS.items():
        lines.append(f"  {name:9s} {desc}")
    return "\n".join(lines)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(usage())
        return
    cmd, rest = argv[0], argv[1:]
    if cmd == "list":
        from neurofly_training.body.tasks import TASKS
        from neurofly_training.data.connectome import SUBSETS
        print("body tasks (the MuJoCo fly; --brain optional):")
        for t in sorted(TASKS):
            print(f"  {t}")
        print("PC tasks (video and sound in, keyboard and mouse out; --brain required):")
        print("  pc             the live screen: --window / --region, --keys, --buttons, --mouse, "
              "--audio, --reward")
        print("  <video file>   a recording, for imitation or for watching the brain react")
        print(f"connectome subsets: {', '.join(sorted(SUBSETS))}")
        return
    if cmd == "devices":
        from neurofly_core.io.audio import list_audio_devices
        print(list_audio_devices())
        return
    if cmd not in COMMANDS:
        raise SystemExit(f"unknown command {cmd!r}\n\n{usage()}")
    module, _ = COMMANDS[cmd]
    sys.argv = [f"neurofly {cmd}"] + rest
    importlib.import_module(f"neurofly_training.cli.{module}").main()
