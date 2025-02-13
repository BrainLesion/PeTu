from __future__ import annotations

import json
import logging
import os
import signal
import sys
import tempfile
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from loguru import logger

from petu.data_handler import DataHandler
from petu.model_handler import ModelHandler


class Inferer:

    def __init__(
        self,
        device: Optional[str] = "cuda",
        cuda_visible_devices: Optional[str] = "0",
    ) -> None:
        self.device = self._configure_device(
            requested_device=device,
            cuda_visible_devices=cuda_visible_devices,
        )
        self.data_handler = DataHandler()
        self.model_handler = ModelHandler(device=self.device)

    def _configure_device(
        self, requested_device: str, cuda_visible_devices: str
    ) -> torch.device:
        """Configure the device for inference based on the specified config.device.

        Returns:
            torch.device: Configured device.
        """
        device = torch.device(requested_device)
        if device.type == "cuda":
            # The env vars have to be set before the first call to torch.cuda, else torch will always attempt to use the first device
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
            if torch.cuda.is_available():
                # clean memory
                torch.cuda.empty_cache()
                logger.debug(
                    f"Set CUDA_VISIBLE_DEVICES to {os.environ['CUDA_VISIBLE_DEVICES']}"
                )
                logger.debug(f"Available CUDA devices: {torch.cuda.device_count()}")
                logger.debug(f"Current CUDA devices: {torch.cuda.current_device()}")

        logger.info(f"Set torch device: {device}")

        return device

    # @citation_reminder
    def infer(
        self,
        t1c: Optional[str | Path | np.ndarray] = None,
        fla: Optional[str | Path | np.ndarray] = None,
        t1: Optional[str | Path | np.ndarray] = None,
        t2: Optional[str | Path | np.ndarray] = None,
        ET_segmentation_file: Optional[str | Path] = None,
        CC_segmentation_file: Optional[str | Path] = None,
        T2H_segmentation_file: Optional[str | Path] = None,
    ) -> Dict[str, np.ndarray]:

        # check inputs and get mode , if mode == prev mode => run inference, else load new model
        validated_images = self.data_handler.validate_images(
            t1=t1, t1c=t1c, t2=t2, fla=fla
        )
        determined_inference_mode = self.data_handler.determine_inference_mode(
            images=validated_images
        )

        self.model_handler.load_model(
            inference_mode=determined_inference_mode,
        )

        with tempfile.TemporaryDirectory() as tmpdir:

            input_file_paths = self.data_handler.get_input_file_paths(
                images=validated_images,
                tmp_folder=Path(tmpdir),
            )

            logger.info(f"Running inference on device: {self.device}")
            np_results = self.model_handler.infer(
                input_file_paths=input_file_paths,
                ET_segmentation_file=ET_segmentation_file,
                CC_segmentation_file=CC_segmentation_file,
                T2H_segmentation_file=T2H_segmentation_file,
            )
            logger.info(f"Finished inference")
            return np_results
