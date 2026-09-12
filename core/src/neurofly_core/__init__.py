"""neurofly-core: the runtime for a trained fly-brain controller.

    Model      the assembled controller: brain + encoders + readout + policy + control layout
    artifact   the on-disk format (manifest.json + flat binary arrays); load_model / save_model
    server     a JSON-lines protocol over stdio or WebSocket, for other languages
    io         video and audio sources, keyboard and mouse output, on the real PC

    import neurofly_core as fc
    model = fc.load_model("artifacts/myapp")
    state, info = model.step(frame, audio)      # ControlState: keys, buttons, dx, dy, scroll

Training lives in neurofly-training; this package has no training code and no data loading.
"""
import warnings

__version__ = "0.1.0"

# torch prints this once per process for every sparse CSR tensor we build
warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta")
warnings.filterwarnings("ignore", message="Sparse invariant checks are implicitly disabled")

from neurofly_core.artifact import load_model, save_model, validate  # noqa: E402,F401
from neurofly_core.controls import ControlLayout, ControlState  # noqa: E402,F401
from neurofly_core.body import BodyModel  # noqa: E402,F401
from neurofly_core.model import Model, ModelConfig  # noqa: E402,F401
