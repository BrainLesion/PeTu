from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from petu.data_handler import DataHandler
from petu.inferer import Inferer
from petu.model_handler import ModelHandler


@pytest.fixture
@patch("petu.inferer.DataHandler")
@patch("petu.inferer.ModelHandler")
def inferer(mock_model_handler, mock_data_handler):
    return Inferer(device="cpu")


@patch("petu.inferer.torch.cuda.is_available", return_value=True)
@patch("petu.inferer.DataHandler")
@patch("petu.inferer.ModelHandler")
def test_device_configuration_gpu(
    mock_model_handler, mock_data_handler, mock_cuda_is_available
):
    inferer_gpu = Inferer(device="cuda", cuda_visible_devices="0")
    assert inferer_gpu.device.type == "cuda"


@patch("petu.inferer.DataHandler")
@patch("petu.inferer.ModelHandler")
def test_device_configuration_cpu(mock_model_handler, mock_data_handler):
    inferer_cpu = Inferer(device="cpu")
    assert inferer_cpu.device.type == "cpu"


def test_infer(inferer):
    inferer.infer(t1c="test")
