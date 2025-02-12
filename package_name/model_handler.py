from __future__ import annotations

import logging
import os
from pathlib import Path
import tempfile
from typing import Dict, List

import numpy as np
import torch
from loguru import logger
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from package_name.constants import WEIGHTS_DIR_PATTERN, InferenceMode

# from brainles_aurora.utils import check_model_weights


class ModelHandler:
    """Class for model loading, inference and post processing"""

    def __init__(self, device: torch.device) -> "ModelHandler":
        self.device = device
        # Will be set during infer() call
        self.predictor = None
        self.inference_mode = None

        # get location of model weights
        self.model_weights_folder = Path()  # check_model_weights()

    def load_model(self, inference_mode: InferenceMode) -> None:

        if not self.predictor or self.inference_mode != inference_mode:
            logger.debug(
                f"No loaded compatible model found (Switching from {self.inference_mode} to {inference_mode}). Loading Model and weights..."
            )
            self.inference_mode = inference_mode
            self.predictor = nnUNetPredictor(
                device=self.device,
            )
            self.predictor.initialize_from_trained_model_folder(
                "/home/marcelrosier/pediatric_test/Dataset500_t1c_fla_t1_t2/nnUNetTrainer__nnUNetPlans__3d_fullres",  # TODO make dependent on inference_mode
                use_folds=("all"),
                checkpoint_name="checkpoint_final.pth",
            )

            logger.debug(f"Successfully loaded model.")
        else:
            logger.debug(
                f"Same inference mode ({self.inference_mode}) as previous infer call. Re-using loaded model"
            )

    def infer(self, input_files: List[Path]) -> Dict[str, np.ndarray]:
        with tempfile.TemporaryDirectory() as tmpdir:
            files = [str(f) for f in input_files]
            self.predictor.predict_from_files(
                [files],
                tmpdir,
                save_probabilities=True,
            )

            print(list(Path(tmpdir).iterdir()))
            # TODO load and threshold npz
            # TODO save segmentations to disk
