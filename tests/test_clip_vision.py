import numpy as np
from diffusers_helper.clip_vision import hf_clip_vision_encode


class _DummyBatch(dict):
    def to(self, **kwargs):
        return self


class _DummyExtractor:
    def preprocess(self, images, return_tensors="pt"):
        assert return_tensors == "pt"
        assert isinstance(images, np.ndarray)
        return _DummyBatch({"pixel_values": "ok"})


class _DummyEncoder:
    device = "cpu"
    dtype = "float32"

    def __call__(self, **kwargs):
        # Return a dummy output to simulate a successful encode
        return {"last_hidden_state": "dummy"}


def test_hf_clip_vision_encode_smoke():
    """Smoke test: hf_clip_vision_encode should call extractor and encoder successfully."""
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    out = hf_clip_vision_encode(img, _DummyExtractor(), _DummyEncoder())
    assert isinstance(out, dict)
    assert "last_hidden_state" in out
