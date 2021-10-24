import sys
import re

import torch
import PIL

from ff_utils import generate_image_crops
from net import Net

from torchvision.ops import nms


def draw_nms_on_image(initial_image, boxes, scores):
    if len(boxes) > 0:
        tensor_boxes = torch.tensor(boxes, dtype=torch.float32)
        tensor_scores = torch.tensor(scores, dtype=torch.float32)

        # normalize
        m, M = min(tensor_scores), max(tensor_scores)
        if len(tensor_scores) > 1:
            tensor_scores = (tensor_scores - m) / (M - m)
        else:
            tensor_scores = tensor_scores * 0 + 1.0

        kept_boxes_idx = nms(
            boxes=tensor_boxes,
            scores=tensor_scores,
            iou_threshold=0.2,
        )
        kept_boxes = [t.tolist() for t in tensor_boxes[kept_boxes_idx]]
        kept_scores = tensor_scores[kept_boxes_idx]
        # kept_scores = kept_scores / max(kept_scores)

        print(list(sorted(scores)))

        res_image = initial_image.copy()
        draw = PIL.ImageDraw.Draw(res_image)
        for (box, score) in zip(kept_boxes, kept_scores):
            # color = "#00ff00"
            # color += hex(round(score.item() * 255)).split('x')[-1].zfill(2)
            color = f'hsl({round(score.item() * 120)}, 100%, 50%)'
            draw.rectangle(box, fill=None, outline=color)
            # font = PIL.ImageFont.truetype('Arial.ttf', 12)
            # draw.text(
            #     (box[0], max(box[1] - 16, 2)),
            #     "score="+str(round(100 * score.item()))+"%",
            #     font=font,
            #     fill=(0, 255, 0, 255)
            # )
        res_image.save('./results/res.png')


if len(sys.argv) < 1:
    print('missing argument: image_path')
    exit(1)

if __name__ == '__main__':
    net = Net()
    net.load_from_file()

    image_path = sys.argv[1]
    initial_image = PIL.Image.open(image_path)

    img_id = re.sub(r'[^a-z0-9]+', '', image_path)

    initial_size = tuple(initial_image.size)
    initial_w, initial_h = initial_size
    crop_w, crop_h = 36, 36
    stride_w, stride_h = crop_w // 2, crop_h // 2

    N_SCALES = 15
    SCALE_FACTOR = (36 / min(initial_w, initial_h)) ** (1 / (N_SCALES + 1))
    image = initial_image.copy().convert('L')

    boxes = []
    scores = []

    min_score = -999

    for d in generate_image_crops(initial_image):
        outputs = net(d.input[None, ...])

        score, predicted = torch.max(outputs.data, 1)
        score = score.item()
        predicted = predicted.item()

        if score > min_score and predicted == 1:
            boxes.append(list(d.rect))
            scores.append(score)

    draw_nms_on_image(initial_image, boxes, scores)
