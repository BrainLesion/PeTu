from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from loguru import logger
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from petu.constants import SEGMENTATION_LABELS, SEGMENTATION_THRESHOLD, InferenceMode
from petu.weights import check_weights_path


class ModelHandler:
    """Class for model loading, inference and post processing"""

    def __init__(self, device: torch.device) -> "ModelHandler":
        """Initialize the ModelHandler class.

        Args:
            device (torch.device): Device to use for inference.

        Returns:
            ModelHandler: ModelHandler instance.
        """

        self.device = device
        # Will be set during infer() call
        self.predictor = None
        self.inference_mode = None

        # get location of model weights
        self.model_weights_folder = check_weights_path()

    def load_model(self, inference_mode: InferenceMode) -> None:
        """Load the model for inference based on the inference mode

        Args:
            inference_mode (InferenceMode): inference mode (determined by passed images)
        """

        if not self.predictor or self.inference_mode != inference_mode:
            logger.debug(
                f"No loaded compatible model found (Switching from {self.inference_mode} to {inference_mode}). Loading Model and weights..."
            )
            self.inference_mode = inference_mode
            self.predictor = nnUNetPredictor(
                device=self.device,
            )
            self.predictor.initialize_from_trained_model_folder(
                self.model_weights_folder / self.inference_mode.value.replace("-", "_"),
                use_folds=("all"),
            )

            logger.debug(f"Successfully loaded model.")
        else:
            logger.debug(
                f"Same inference mode ({self.inference_mode}) as previous infer call. Re-using loaded model"
            )

    def threshold_probabilities(
        self, results_dir: Path
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Threshold the probabilities to get binary segmentations.

        Args:
            results_dir (Path): output directory of the nnUNet inference.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple of binary segmentations (ET, CC, T2H) and affine matrix.
        """

        nifti_file = next(results_dir.glob("*.nii.gz"))
        affine = nib.load(nifti_file).affine

        probabilities_file = next(results_dir.glob("*.npz"))
        probabilities = np.load(probabilities_file)["probabilities"]

        binary_data_list = []
        for i, _ in enumerate(SEGMENTATION_LABELS):
            transposed_data = np.transpose(probabilities[i], (2, 1, 0))
            binary_data = np.where(
                transposed_data > SEGMENTATION_THRESHOLD, 1, 0
            ).astype(np.int8)
            binary_data_list.append(binary_data)
        return *binary_data_list, affine

    def infer(
        self,
        input_file_paths: List[Path],
        ET_segmentation_file: Optional[str | Path] = None,
        CC_segmentation_file: Optional[str | Path] = None,
        T2H_segmentation_file: Optional[str | Path] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run inference on the provided images and save the segmentations to disk if paths are provided.

        Args:
            input_file_paths (List[Path]): _description_
            ET_segmentation_file (Optional[str  |  Path], optional): Path to save ET segmentation. Defaults to None.
            CC_segmentation_file (Optional[str  |  Path], optional): Path to save CC segmentation. Defaults to None.
            T2H_segmentation_file (Optional[str  |  Path], optional): Path to save T2H segmentation. Defaults to None.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple of segmentations: (ET, CC, T2H) as numpy arrays.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            str_paths = [str(f) for f in input_file_paths]
            self.predictor.predict_from_files(
                [str_paths],
                tmpdir,
                save_probabilities=True,
            )
            et, cc, t2h, affine = self.threshold_probabilities(
                results_dir=Path(tmpdir),
            )

            # save segmentations to disk
            for data, path in zip(
                [et, cc, t2h],
                [ET_segmentation_file, CC_segmentation_file, T2H_segmentation_file],
            ):
                if path is not None:
                    path = Path(path)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    nib.save(
                        nib.Nifti1Image(data, affine),
                        path,
                    )
                    logger.debug(f"Saved segmentation to {path}")

            return et, cc, t2h
