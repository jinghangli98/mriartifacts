"""mriartifacts: generate realistic MRI artifacts (motion, spike, GRAPPA, low resolution)
on clean brain volumes with a class-conditioned flow-matching model.

The model was trained with five degradation classes (the class-table order
below); the CLI exposes the three validated in the ISBI 2027 paper plus grappa
(unvalidated); cs stays API-only.
Constants only here: submodules import them, so this file must not import
submodules (circular import)."""

__version__ = "0.1.0"

FAMILIES = ("motion", "spike", "grappa", "lowres", "cs")   # class-table order of the trained model
NULL_INDEX = len(FAMILIES)                                  # unconditional row for classifier-free guidance
NUM_CLASSES = len(FAMILIES) + 1
PUBLIC_ARTIFACTS = ("motion", "spike", "grappa", "lowres")  # the names the CLI accepts (grappa unvalidated)
