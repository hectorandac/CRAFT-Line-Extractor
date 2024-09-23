import os, sys
import pytest
import torch
import numpy as np
from unittest import mock

repo_path = os.path.join(os.path.dirname(__file__), '../CRAFT')
sys.path.append(repo_path)

from CRAFT.craft import CRAFT
from CRAFT import craft_utils, imgproc
import craft_text_detector
from craft_text_detector import load_craft_model, copy_state_dict, detect_text_from_image

@pytest.fixture(autouse=True)
def reset_mocks():
    """Automatically reset mocks between tests."""
    mock.patch.stopall()
    yield
    mock.patch.stopall()

def test_copy_state_dict():
    state_dict_with_module = {"module.layer1.weight": torch.tensor([1, 2]), "module.layer2.bias": torch.tensor([3, 4])}
    copied_state_dict = copy_state_dict(state_dict_with_module)
    assert "layer1.weight" in copied_state_dict
    assert copied_state_dict["layer1.weight"].equal(torch.tensor([1, 2]))

    state_dict_no_module = {"layer1.weight": torch.tensor([1, 2]), "layer2.bias": torch.tensor([3, 4])}
    copied_state_dict = copy_state_dict(state_dict_no_module)
    assert "layer1.weight" in copied_state_dict
    assert copied_state_dict["layer1.weight"].equal(torch.tensor([1, 2]))

@mock.patch("craft_text_detector.CRAFT", autospec=True)
@mock.patch("craft_text_detector.torch.load")
def test_load_craft_model(mock_torch_load, mock_CRAFT):
    # Reset the global variable `craft_model` to ensure a clean test state
    craft_text_detector.craft_model = None
    
    mock_model = mock.MagicMock()
    mock_CRAFT.return_value = mock_model
    mock_torch_load.return_value = {"mock_key": "mock_value"}

    model = load_craft_model("path/to/model.pth", use_cuda=False)
    
    mock_CRAFT.assert_called_once()
    mock_model.load_state_dict.assert_called_once_with({"mock_key": "mock_value"})
    assert model is not None

@mock.patch("craft_text_detector.craft_utils.getDetBoxes")
@mock.patch("craft_text_detector.imgproc.resize_aspect_ratio")
@mock.patch("craft_text_detector.imgproc.normalizeMeanVariance")
@mock.patch("craft_text_detector.load_craft_model")
def test_detect_text_from_image(mock_load_craft_model, mock_normalizeMeanVariance, mock_resize_aspect_ratio, mock_getDetBoxes):
    mock_model = mock.MagicMock()
    mock_load_craft_model.return_value = mock_model

    mock_image = torch.rand(3, 128, 128).numpy()
    mock_resize_aspect_ratio.return_value = (torch.rand(1280, 1280, 3).numpy(), 1.0, 1.0)
    mock_normalizeMeanVariance.return_value = torch.rand(1280, 1280, 3).numpy()
    mock_model.return_value = (torch.rand(1, 1280, 1280, 2), None)
    mock_getDetBoxes.return_value = ([], [])

    boxes, polys, score_text = detect_text_from_image(mock_image, use_cuda=False)

    mock_load_craft_model.assert_called_once()
    mock_resize_aspect_ratio.assert_called_once()
    mock_normalizeMeanVariance.assert_called_once()

    assert isinstance(boxes, list)
    assert isinstance(polys, list)
    assert isinstance(score_text, np.ndarray)
