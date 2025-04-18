bucket_options = {
    240: [
        (224, 336),
        (256, 304),
        (288, 272),
        (320, 240),
        (352, 208),
        (384, 176),
    ],
    320: [
        (256, 576),
        (288, 544),
        (320, 512),
        (352, 480),
        (384, 448),
        (416, 416),
        (448, 384),
        (480, 352),
        (512, 320),
        (544, 288),
        (576, 256),
    ],
    640: [
        (416, 960),
        (448, 864),
        (480, 832),
        (512, 768),
        (544, 704),
        (576, 672),
        (608, 640),
        (640, 608),
        (672, 576),
        (704, 544),
        (768, 512),
        (832, 480),
        (864, 448),
        (960, 416),
    ],
}


def find_nearest_bucket(h, w, resolution=640):
    min_metric = float("inf")
    best_bucket = None
    for bucket_h, bucket_w in bucket_options[resolution]:
        metric = abs(h * bucket_w - w * bucket_h)
        if metric <= min_metric:
            min_metric = metric
            best_bucket = (bucket_h, bucket_w)
    print(f"Best bucket: {best_bucket}")
    return best_bucket
