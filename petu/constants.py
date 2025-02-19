from enum import Enum
from pathlib import Path


class InferenceMode(str, Enum):
    """
    Enum representing different modes of inference based on available image inputs.
    In general, you should aim to use as many modalities as possible to get the best results.
    """

    T1C_FLA_T1_T2 = "t1c-fla-t1-t2"
    """T1C, FLAIR, T1, and T2 are available."""

    T1C = "t1c"
    """T1C is available."""

    FLA = "fla"
    """FLAIR is available."""

    T1 = "t1"
    """T1 is available."""

    T2 = "t2"
    """T2 is available."""

    T1C_FLA = "t1c-fla"
    """T1C and FLAIR are available."""

    T1C_T1 = "t1c-t1"
    """T1C and T1 are available."""

    T1C_T1_T2 = "t1c-t1-t2"
    """T1C, T1, and T2 are available."""

    T1C_T2 = "t1c-t2"
    """T1C and T2 are available."""

    FLA_T1 = "fla-t1"
    """FLAIR and T1 are available."""

    FLA_T1_T2 = "fla-t1-t2"
    """FLAIR, T1, and T2 are available."""

    FLA_T2 = "fla-t2"
    """FLAIR and T2 are available."""


class DataMode(str, Enum):
    """Enum representing different modes for handling input and output data."""

    NIFTI_FILE = "NIFTI_FILEPATH"
    """Input data is provided as NIFTI file paths/ output is written to NIFTI files."""
    NUMPY = "NP_NDARRAY"
    """Input data is provided as NumPy arrays/ output is returned as NumPy arrays."""


MODALITIES = ["t1c", "fla", "t1", "t2"]
"""List of modality names in standard order: T1C, FLAIR, T1, T2."""


# booleans indicate presence of files in order: T1C, FLAIR, T1, T2
IMGS_TO_MODE_DICT = {
    (True, True, True, True): InferenceMode.T1C_FLA_T1_T2,
    (True, False, False, False): InferenceMode.T1C,
    (False, True, False, False): InferenceMode.FLA,
    (False, False, True, False): InferenceMode.T1,
    (False, False, False, True): InferenceMode.T2,
    (True, True, False, False): InferenceMode.T1C_FLA,
    (True, False, True, False): InferenceMode.T1C_T1,
    (True, False, True, True): InferenceMode.T1C_T1_T2,
    (True, False, False, True): InferenceMode.T1C_T2,
    (False, True, True, False): InferenceMode.FLA_T1,
    (False, True, True, True): InferenceMode.FLA_T1_T2,
    (False, True, False, True): InferenceMode.FLA_T2,
}
"""Dictionary mapping tuples of booleans representing presence of the modality in order [T1C, FLAIR, T1, T2] to InferenceMode values."""

ZENODO_RECORD_URL = "https://zenodo.org/api/records/14864230"
WEIGHTS_FOLDER = Path(__file__).parent / "weights"
WEIGHTS_DIR_PATTERN = "weights_v*.*.*"
"""Directory name pattern to store model weights. E.g. weights_v1.0.0"""

ATLAS_SPACE_SHAPE = (240, 240, 155)
"""Standard shape of the atlas space."""

SEGMENTATION_THRESHOLD = 0.5
SEGMENTATION_LABELS = ["ET", "CC", "T2H"]

NNUNET_ENV_VARS = [
    "nnUNet_raw",
    "nnUNet_preprocessed",
    "nnUNet_results",
]
