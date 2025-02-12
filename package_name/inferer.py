from __future__ import annotations

import json
import logging
import os
import signal
import sys
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from package_name.data_handler import DataHandler
from loguru import logger


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
        # self.model_handler = ModelHandler(config=self.config, device=self.device)

    def _configure_device(
        self, requested_device: str, cuda_visible_devices: str
    ) -> torch.device:
        """Configure the device for inference based on the specified config.device.

        Returns:
            torch.device: Configured device.
        """
        device = torch.device(requested_device)
        if device.type == "cuda":
            # The env vars have to be set ebfore the first call to torch.cuda, else torch will always attempt to use the first device
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
        t1c: str | Path | np.ndarray | None = None,
        fla: str | Path | np.ndarray | None = None,
        t1: str | Path | np.ndarray | None = None,
        t2: str | Path | np.ndarray | None = None,
        CC_segmentation_file: str | Path | None = None,
        ET_segmentation_file: str | Path | None = None,
        T2H_segmentation_file: str | Path | None = None,
    ) -> Dict[str, np.ndarray]:

        # check inputs and get mode , if mode == prev mode => run inference, else load new model
        validated_images = self.data_handler.validate_images(
            t1=t1, t1c=t1c, t2=t2, fla=fla
        )
        determined_inference_mode = self.data_handler.determine_inference_mode(
            images=validated_images
        )

        # self.model_handler.load_model(
        #     inference_mode=determined_inference_mode,
        #     num_input_modalities=self.data_handler.get_num_input_modalities(),
        # )

        logger.info(f"Running inference on device := {self.device}")
        # out = self.model_handler.infer(data_loader=data_loader)
        logger.info(f"Finished inference")

        # save data to fie if paths are provided
        # if any(output_file_mapping.values()):
        #     logger.info("Saving post-processed data as NIfTI files")
        #     self.data_handler.save_as_nifti(
        #         postproc_data=out, output_file_mapping=output_file_mapping
        #     )
        # logger.info(f"{' Finished inference run ':=^80}")
        # return out
