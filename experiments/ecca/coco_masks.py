"""COCO diagnostic masks using existing NumPy, without pycocotools.

Polygon rasterization and compressed-RLE decoding are Python translations of
COCO API's rleFrPoly/rleFrString (Piotr Dollar and Tsung-Yi Lin, 2015).
Source: https://github.com/cocodataset/cocoapi/blob/master/common/maskApi.c
See COCO_API_LICENSE.txt. This is a mask decoder, not a COCO evaluator.
"""
import math
from numbers import Integral
import numpy as np

BACKEND = 'numpy_coco_rle_v1'


def _counts_from_string(text):
    if isinstance(text, bytes):
        text = text.decode('ascii')
    counts = []
    position = 0
    while position < len(text):
        value = shift = 0
        while True:
            if position >= len(text):
                raise ValueError('Truncated COCO compressed RLE')
            code = ord(text[position]) - 48
            position += 1
            if not 0 <= code <= 63:
                raise ValueError('Invalid COCO compressed RLE character')
            value |= (code & 31) << shift
            shift += 5
            if not code & 32:
                if code & 16:
                    value |= -1 << shift
                break
            if shift > 65:
                raise ValueError('Invalid COCO compressed RLE integer')
        if len(counts) > 2:
            value += counts[-2]
        counts.append(value)
    return counts


def _decode_counts(counts, height, width):
    if not isinstance(counts, (list, tuple)) or any(
            not isinstance(n, Integral) or n < 0 for n in counts):
        raise ValueError('COCO RLE counts must be nonnegative integers')
    if sum(counts) != height * width:
        raise ValueError('COCO RLE run total does not match image size')
    flat = np.zeros(height * width, dtype=bool)
    offset = 0
    for index, length in enumerate(counts):
        if index % 2:
            flat[offset:offset + length] = True
        offset += length
    return flat.reshape((height, width), order='F')


def _polygon_counts(polygon, height, width):
    coords = np.asarray(polygon, dtype=float)
    if coords.ndim != 1 or len(coords) < 6 or len(coords) % 2 or not np.isfinite(coords).all():
        raise ValueError('COCO polygon needs at least three finite (x,y) vertices')
    # int() intentionally follows C truncation rather than Python round().
    x = [int(5 * value + .5) for value in coords[0::2]]
    y = [int(5 * value + .5) for value in coords[1::2]]
    x.append(x[0]); y.append(y[0])
    boundary = []
    for index in range(len(x) - 1):
        xs, xe, ys, ye = x[index], x[index + 1], y[index], y[index + 1]
        dx, dy = abs(xe - xs), abs(ye - ys)
        if dx == dy == 0:
            boundary.append((xs, ys))
            continue
        flip = (dx >= dy and xs > xe) or (dx < dy and ys > ye)
        if flip:
            xs, xe, ys, ye = xe, xs, ye, ys
        if dx >= dy:
            slope = (ye - ys) / dx
            for distance in range(dx + 1):
                t = dx - distance if flip else distance
                boundary.append((xs + t, int(ys + slope * t + .5)))
        else:
            slope = (xe - xs) / dy
            for distance in range(dy + 1):
                t = dy - distance if flip else distance
                boundary.append((int(xs + slope * t + .5), ys + t))
    points = []
    for (u0, v0), (u1, v1) in zip(boundary, boundary[1:]):
        if u0 == u1:
            continue
        xd = ((u1 if u1 < u0 else u1 - 1) + .5) / 5 - .5
        if math.floor(xd) != xd or not 0 <= xd <= width - 1:
            continue
        yd = (min(v0, v1) + .5) / 5 - .5
        points.append(int(xd) * height + math.ceil(min(height, max(0, yd))))
    points.append(height * width)
    points.sort()
    deltas = [points[0]] + [b - a for a, b in zip(points, points[1:])]
    counts, index = [deltas[0]], 1
    while index < len(deltas):
        if deltas[index] > 0:
            counts.append(deltas[index]); index += 1
        else:
            index += 1
            if index < len(deltas):
                counts[-1] += deltas[index]; index += 1
    return counts


def segmentation_mask(segmentation, height, width):
    """Decode polygons (union), uncompressed RLE, or compressed COCO RLE."""
    if not isinstance(height, Integral) or not isinstance(width, Integral) or min(height, width) <= 0:
        raise ValueError('COCO image dimensions must be positive integers')
    if isinstance(segmentation, list):
        mask = np.zeros((height, width), dtype=bool)
        for polygon in segmentation:
            mask |= _decode_counts(_polygon_counts(polygon, height, width), height, width)
        return mask
    if not isinstance(segmentation, dict) or 'counts' not in segmentation:
        raise ValueError('Expected COCO polygons or RLE segmentation; bbox is not a mask')
    if list(segmentation.get('size', [])) != [height, width]:
        raise ValueError('COCO RLE size differs from image dimensions')
    counts = segmentation['counts']
    if isinstance(counts, (str, bytes)):
        counts = _counts_from_string(counts)
    return _decode_counts(counts, height, width)


def semantic_mask(annotations, height, width, category_lookup):
    """Keep prior ECCA policy: all overlaps/crowds -> 255; unannotated -> 0."""
    if not category_lookup or max(category_lookup.values()) >= 255:
        raise ValueError('Diagnostic class IDs must be in 1..254')
    gt = np.zeros((height, width), dtype=np.uint8)
    occupied = np.zeros_like(gt, dtype=bool)
    invalid = np.zeros_like(gt, dtype=bool)
    for annotation in annotations:
        mask = segmentation_mask(annotation['segmentation'], height, width)
        invalid |= mask & occupied
        if annotation.get('iscrowd', 0):
            invalid |= mask
        gt[mask] = category_lookup[int(annotation['category_id'])]
        occupied |= mask
    gt[invalid] = 255
    return gt
