import dataclasses

import PIL
import torch
import torchvision.transforms as transforms
from torchvision.ops import nms

transform = transforms.Compose(
    [transforms.Grayscale(),
     transforms.ToTensor(),
     transforms.Normalize(mean=(0,), std=(1,))])


def pyramid(image, scale=1.5, minSize=(30, 30)):
    # yield the original image
    yield image
    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.size[0] / scale)
        h = int(image.size[1] / scale)
        image.thumbnail((w, h), PIL.Image.LANCZOS)

        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.size[0] < minSize[0] or image.size[1] < minSize[1]:
            break
        # yield the next image in the pyramid
        yield image


def sliding_window(image, stepSize, windowSize):
    winW, winH = windowSize
    # slide a window across the image
    for x in range(0, image.size[0], stepSize):
        for y in range(0, image.size[1], stepSize):
            # yield the current window
            left, top, right, bottom = x, y, x + winW, y + winH
            cropped_image = image.crop((left, top, right, bottom))
            t = transform(cropped_image)
            box = (left, top, right, bottom)
            yield (box, t)


def reshape_scale_box(tlrb, zw, zh):
    """
    scale the (left, top, right, bottom) tuple,
    reverting the scale factor (zw, zh)
    """
    l, t, r, b = tlrb
    l = zw * l
    t = zh * t
    r = zw * r
    b = zh * b
    l, t, r, b = round(l), round(t), round(r), round(b)
    return (l, t, r, b)


def compute_nms(boxes, scores, iou_threshold=0.2):
    if len(boxes) == 0:
        return [], []

    tensor_boxes = torch.tensor(boxes, dtype=torch.float32)
    tensor_scores = torch.tensor(scores, dtype=torch.float32)

    # normalize
    # m, M = min(tensor_scores), max(tensor_scores)
    # if len(tensor_scores) > 1:
    #     tensor_scores = (tensor_scores - m) / (M - m)
    # else:
    #     tensor_scores = tensor_scores * 0 + 1.0

    kept_boxes_idx = nms(
        boxes=tensor_boxes,
        scores=tensor_scores,
        iou_threshold=iou_threshold,
    )
    kept_boxes = [tuple(t.tolist()) for t in tensor_boxes[kept_boxes_idx]]
    kept_scores = [s.item() for s in tensor_scores[kept_boxes_idx]]
    return kept_boxes, kept_scores


def pyramid_sliding_window_detection(net, image, scale, winW, winH, stepSize, minScore=0.9):
    # Store the initial image before resize, it will be used for the final printing
    image_copy = image.copy()

    initial_size = image_copy.size

    boxes, scores = [], []
    for resized in pyramid(image, scale=scale, minSize=(winW, winH)):
        print(resized.size[0], '×', resized.size[1])
        zw = initial_size[0] / resized.size[0]
        zh = initial_size[1] / resized.size[1]
        # loop over the sliding window for each layer of the pyramid

        win = sliding_window(
            resized, stepSize=stepSize, windowSize=(winW, winH))

        for (box, t) in win:
            # We use the 36*36 window to match the net's img input size
            # Feed the network the input tensor
            output = net(t[None, ...])

            # We only register faces with a prob higher than {minScore} to avoid false positives
            # (softmax dim parameter : dim=0->rows add up to 1, dim=1->rows add up to 1)
            softmax = torch.nn.functional.softmax(output, dim=1)
            scoreF = softmax[0][1]
            if scoreF >= minScore:
                box = reshape_scale_box(box, zw, zh)
                boxes.append(box)
                scores.append(scoreF)

    # run non max suppression
    boxes, scores = compute_nms(boxes, scores)

    return boxes, scores


@dataclasses.dataclass
class FaceFinderCrop:
    rect: tuple
    input: object
    id: str
    img: PIL
    z: int


def generate_image_crops(
    initial_image,
    scale_factor: int = 'Auto',
    n_scales: int = 15,
):
    initial_w, initial_h = initial_image.size
    crop_w, crop_h = 36, 36
    stride_w, stride_h = crop_w // 2, crop_h // 2

    if scale_factor == 'Auto':
        scale_factor = auto_scale(n_scales, initial_w, initial_h)

    image = initial_image.copy().convert('L')

    for z in range(n_scales):
        if z > 0:  # resize source image
            newsize = (initial_w * (scale_factor ** (z + 1)),
                       initial_h * (scale_factor ** (z + 1)))
            image.thumbnail(newsize, PIL.Image.LANCZOS)

        W, H = image.size

        print(f'searching at {W}×{H}')

        if W < crop_w or H < crop_h:
            print('break early', W, H)
            break

        # stride_w = int(min(
        #     (W / math.ceil(W/36)) // 2,
        #     crop_w // 2
        # ))
        # stride_h = int(min(
        #     (H / math.ceil(H/36)) // 2,
        #     crop_h // 2
        # ))

        for x in range(0, W - crop_w, stride_w):
            for y in range(0, H - crop_h, stride_h):
                left, top, right, bottom = x, y, x+crop_w, y+crop_h

                cropped_image = image.crop((left, top, right, bottom))
                t = transform(cropped_image)

                fx, fy = initial_w * left/W, initial_h * top/H
                fw = initial_w * (right - left) / W
                fh = initial_h * (bottom - top) / H
                fx, fy, fw, fh = round(fx), round(fy), round(fw), round(fh)

                yield FaceFinderCrop(
                    rect=(fx, fy, fx+fw, fy+fh),
                    input=t,
                    id=f'{left}-{top}-{right}-{bottom}-{z}',
                    img=cropped_image,
                    z=z,
                )


def auto_scale(n_scales, initial_w, initial_h):
    """
    We might want to detect portrait pictures (1 face per image),
    so the scaling should go to ~36x36 (with correct aspect ratio).
    We solve the following equation, for both width and height:

    initial_w * scale_factor ** n_scales = 36

    <-> scale_factor = (36 / initial_w) ** (1 / n_scales)

    and select the biggest value.
    """
    return (36 / min(initial_w, initial_h)) ** (1 / (n_scales + 1))
