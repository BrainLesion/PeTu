from pathlib import Path
from unittest.mock import MagicMock, patch
from unittest.mock import call

import numpy as np
import pytest
import torch

from petu.constants import InferenceMode
from petu.model_handler import ModelHandler


@pytest.fixture
@patch(
    "petu.model_handler.check_weights_path",
    return_value=Path("/mocked/path/to/weights"),
)
def mock_model_handler(_):
    """Fixture to create a ModelHandler instance with a mocked device."""
    device = torch.device("cpu")
    handler = ModelHandler(device)
    return handler


@patch("petu.model_handler.nnUNetPredictor")
def test_load_model(mock_predictor_class, mock_model_handler):
    """Test if load_model correctly initializes the nnUNetPredictor."""
    mock_predictor = MagicMock()
    mock_predictor_class.return_value = mock_predictor

    inference_mode = InferenceMode.T1
    mock_model_handler.load_model(inference_mode)

    assert mock_model_handler.predictor is not None
    # mock_predictor.initialize_from_trained_model_folder
    mock_predictor.initialize_from_trained_model_folder.assert_called_once_with(
        Path("/mocked/path/to/weights") / inference_mode.value.replace("-", "_"),
        use_folds=("all"),
    )


@patch("petu.model_handler.Path.glob")
@patch("petu.model_handler.nib.load")
@patch("petu.model_handler.np.load")
def test_threshold_probabilities(
    mock_np_load, mock_nib_load, mock_glob, mock_model_handler, tmp_path
):
    """Test if threshold_probabilities correctly processes probability maps."""

    # Mock the NIfTI file path returned by glob
    mock_file = tmp_path / "mock_file"
    mock_glob.return_value = iter([mock_file] * 2)

    mock_nifti = MagicMock()
    mock_nifti.affine = np.eye(4)
    mock_nib_load.return_value = mock_nifti

    mock_probabilities = np.random.rand(
        3, 11, 22, 33
    )  # Simulating 3-label probabilities
    mock_np_load.return_value = {"probabilities": mock_probabilities}

    (et, cc, t2h, affine) = mock_model_handler.threshold_probabilities(tmp_path)

    assert et.shape == (33, 22, 11)
    assert cc.shape == (33, 22, 11)
    assert t2h.shape == (33, 22, 11)
    assert np.array_equal(affine, np.eye(4))


@patch("petu.model_handler.nnUNetPredictor")
@patch("petu.model_handler.nib.save")
def test_infer_not_saving(
    mock_nib_save, mock_predictor_class, mock_model_handler, tmp_path
):
    """Test if infer runs inference and saves output correctly."""
    mock_predictor = MagicMock()
    mock_predictor_class.return_value = mock_predictor

    mock_model_handler.predictor = mock_predictor
    input_files = [tmp_path / "image1.nii.gz", tmp_path / "image2.nii.gz"]

    with patch.object(
        mock_model_handler,
        "threshold_probabilities",
        return_value=(np.zeros((10, 10, 10)),) * 3 + (np.eye(4),),
    ):
        et, cc, t2h = mock_model_handler.infer(input_files)

    assert et.shape == (10, 10, 10)
    assert cc.shape == (10, 10, 10)
    assert t2h.shape == (10, 10, 10)
    mock_nib_save.assert_not_called()


@patch("petu.model_handler.nnUNetPredictor")
@patch("petu.model_handler.nib.Nifti1Image")
@patch("petu.model_handler.nib.save")
def test_infer_saving(
    mock_nib_save,
    mock_create_nifti_img,
    mock_predictor_class,
    mock_model_handler,
    tmp_path,
):
    """Test if infer runs inference and saves output correctly."""
    mock_predictor = MagicMock()
    mock_predictor_class.return_value = mock_predictor

    mock_nifti_img = MagicMock()
    mock_create_nifti_img.return_value = mock_nifti_img

    mock_model_handler.predictor = mock_predictor
    input_files = [tmp_path / "image1.nii.gz", tmp_path / "image2.nii.gz"]

    with patch.object(
        mock_model_handler,
        "threshold_probabilities",
        return_value=(np.zeros((10, 10, 10)),) * 3 + (np.eye(4),),
    ):
        et_path = "et.nii.gz"
        t2h_path = "t2h.nii.gz"
        et, cc, t2h = mock_model_handler.infer(
            input_files,
            ET_segmentation_file=et_path,
            CC_segmentation_file=None,
            T2H_segmentation_file=t2h_path,
        )

    assert et.shape == (10, 10, 10)
    assert cc.shape == (10, 10, 10)
    assert t2h.shape == (10, 10, 10)
    mock_nib_save.assert_has_calls(
        [
            call(mock_nifti_img, Path(et_path)),
            call(mock_nifti_img, Path(t2h_path)),
        ]
    )
