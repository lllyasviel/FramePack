# tests/test_bucket_table.py
from diffusers_helper.bucket_tools import bucket_options


def test_bucket_options_nonempty():
    """bucket_options should have integer keys and non-empty lists of (h, w) tuples."""
    assert isinstance(bucket_options, dict)
    assert all(isinstance(k, int) for k in bucket_options)
    assert any(bucket_options.values())  # at least one non-empty list
    for key, lst in bucket_options.items():
        for h, w in lst:
            assert isinstance(h, int)
            assert isinstance(w, int)
            assert h > 0 and w > 0
