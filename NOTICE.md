# Third-party notices

neurofly-kit is AGPL-3.0 (see LICENSE). It builds on the following, each under its own
terms; a project using the kit should keep these credits visible where the data is shown
(the bundled viewers do).

## MaleCNS v1.0 connectome (CC BY 4.0)

The neurons, synapses, annotations and soma positions come from the MaleCNS v1.0 flat
connectome released by the FlyEM Project Team at HHMI Janelia Research Campus, with the
MRC Laboratory of Molecular Biology, Cambridge, and Google Research, under the Creative
Commons Attribution 4.0 licence. `neurofly download` fetches it from Janelia's public
bucket; nothing of it is redistributed in this repository.

## The leaky integrate-and-fire model (Shiu et al., 2024)

The neuron and synapse parameters follow Philip K. Shiu et al., "A Drosophila
computational brain model reveals sensorimotor processing", Nature 634, 210–219 (2024).

## flybody (Apache-2.0)

The fruit fly body model and its tasks are `flybody` by Roman Vaxenburg et al. at
HHMI Janelia and Google DeepMind, Apache-2.0, installed through the `flybody` extra and
exported by `neurofly export-body`.

## Detector backends

The open-vocabulary detectors (OWLv2, Grounding DINO), RT-DETRv2 and D-FINE are used
through the `transformers` package (Apache-2.0) with their own model licences (Apache-2.0);
SSDLite through `torchvision` (BSD-3-Clause); YOLO11 and YOLO-World through `ultralytics`
(AGPL-3.0). `neurofly detect-list` prints the licence of each backend.

## Three.js (MIT)

The bundled viewers load Three.js from a CDN.
