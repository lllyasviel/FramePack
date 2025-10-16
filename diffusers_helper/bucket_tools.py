bucket_options = {
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
}


def find_nearest_bucket(h, w, resolution=640):
    min_metric = float('inf')
    best_bucket = None
    for (bucket_h, bucket_w) in bucket_options:
        metric = abs(h * bucket_w - w * bucket_h)
        if metric <= min_metric:
            min_metric = metric
            best_bucket = (bucket_h, bucket_w)

    if resolution != 640:
        scale_factor = resolution / 640.0
        scaled_height = round(best_bucket[0] * scale_factor / 16) * 16
        scaled_width = round(best_bucket[1] * scale_factor / 16) * 16
        best_bucket = (scaled_height, scaled_width)
        print(f'Resolution: {best_bucket[1]} x {best_bucket[0]}')

    return best_bucket

