import numpy as np
import pytest

from backend.preprocessing import (
    IMAGENET_MEAN,
    circular_crop,
    crop_to_retina,
    looks_like_fundus,
    pad_to_square,
    preprocess_fundus,
    preprocessing_stages,
    retina_mask,
    to_model_tensor,
)


@pytest.fixture
def fundus():

    import cv2

    image = np.zeros((600, 800, 3), dtype=np.uint8)
    cv2.circle(image, (400, 300), 260, (30, 70, 160), -1)
    return image


def test_retina_mask_finds_the_disc_and_not_the_border(fundus):
    mask = retina_mask(fundus)
    assert mask[300, 400], "the centre of the disc should be inside the mask"
    assert not mask[5, 5], "the black corner should be outside the mask"


def test_crop_removes_the_letterbox(fundus):
    cropped = crop_to_retina(fundus)
    assert cropped.shape[0] < fundus.shape[0]
    assert cropped.shape[1] < fundus.shape[1]
    # the disc has a 520 px diameter, so the crop should be close to that
    assert abs(cropped.shape[0] - 521) <= 6
    assert abs(cropped.shape[1] - 521) <= 6


def test_crop_survives_an_all_black_frame():
    black = np.zeros((100, 120, 3), dtype=np.uint8)
    assert crop_to_retina(black).shape == black.shape


def test_padding_squares_without_stretching():
    tall = np.full((300, 100, 3), 200, dtype=np.uint8)
    squared = pad_to_square(tall)
    assert squared.shape[:2] == (300, 300)
    # the original content is centred, not scaled
    assert squared[150, 150].tolist() == [200, 200, 200]
    assert squared[150, 5].tolist() == [0, 0, 0]


def test_circular_mask_blacks_out_the_corners():
    solid = np.full((200, 200, 3), 255, dtype=np.uint8)
    masked = circular_crop(solid)
    assert masked[100, 100].mean() == 255, "the centre survives"
    assert masked[2, 2].mean() == 0, "the corner is masked out"


def test_pipeline_output_shape_and_type(fundus):
    processed = preprocess_fundus(fundus, size=384)
    assert processed.shape == (384, 384, 3)
    assert processed.dtype == np.uint8


@pytest.mark.parametrize("size", [224, 320, 384, 512])
def test_pipeline_respects_the_requested_size(fundus, size):
    assert preprocess_fundus(fundus, size=size).shape == (size, size, 3)


def test_every_stage_is_returned_at_the_model_size(fundus):
    stages = preprocessing_stages(fundus, size=384)
    expected = {"raw", "cropped", "resized", "clahe", "illumination", "final"}
    assert set(stages) == expected
    for name, image in stages.items():
        assert image.shape == (384, 384, 3), f"stage '{name}' has the wrong shape"


def test_illumination_correction_flattens_a_brightness_gradient():
    """The whole point of the Graham step: a lit-from-one-side image evens out."""
    import cv2

    gradient = np.tile(np.linspace(20, 230, 400, dtype=np.uint8), (400, 1))
    image = cv2.merge([gradient, gradient, gradient])
    cv2.circle(image, (200, 200), 180, (0, 0, 0), thickness=0)  # keep it bright

    before = float(image[:, :, 1].astype(np.float32).std())
    after = float(preprocess_fundus(image, size=384)[:, :, 1].astype(np.float32).std())
    assert after < before, "the left-to-right gradient should be reduced"


def test_tensor_is_nchw_float32_and_imagenet_normalised(fundus):
    tensor = to_model_tensor(preprocess_fundus(fundus, 384), 384)
    assert tensor.shape == (1, 3, 384, 384)
    assert tensor.dtype == np.float32

    # a mid-grey pixel should map close to (0.5 - mean) / std
    grey = np.full((384, 384, 3), 128, dtype=np.uint8)
    grey_tensor = to_model_tensor(grey, 384)
    expected_red = (128 / 255 - IMAGENET_MEAN[0]) / 0.229
    assert np.isclose(grey_tensor[0, 0, 10, 10], expected_red, atol=1e-4)


def test_tensor_resizes_a_mismatched_input():
    odd = np.full((250, 250, 3), 90, dtype=np.uint8)
    assert to_model_tensor(odd, 384).shape == (1, 3, 384, 384)


def test_fundus_guard_accepts_a_fundus_and_rejects_a_flat_image(fundus):
    assert looks_like_fundus(fundus)
    assert not looks_like_fundus(np.zeros((400, 400, 3), dtype=np.uint8))
    assert not looks_like_fundus(np.full((400, 400, 3), 240, dtype=np.uint8))
