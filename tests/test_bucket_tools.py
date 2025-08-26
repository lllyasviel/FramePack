from diffusers_helper.bucket_tools import find_nearest_bucket, bucket_options


def test_returns_known_bucket_and_shape():
    """If input matches one of the bucket options exactly, it should return it."""
    h, w = 480, 832
    bh, bw = find_nearest_bucket(h, w, resolution=640)
    assert (bh, bw) in bucket_options[640]
    assert (bh, bw) == (480, 832)


def test_picks_minimum_area_difference():
    """If input does not match exactly, pick the closest area bucket."""
    bh, bw = find_nearest_bucket(500, 800, resolution=640)
    assert (bh, bw) == (512, 768)
