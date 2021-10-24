import sys

import PIL

from net import Net
from ff_utils import pyramid_sliding_window_detection, auto_scale


def draw_boxes_on_image(initial_image, boxes, scores):
    out_path = sys.argv[2] if len(sys.argv) >= 2 else './found_faces.png'

    res_image = initial_image.copy()
    draw = PIL.ImageDraw.Draw(res_image)
    for (box, score) in zip(boxes, scores):
        color = f'hsl({round(score * 120)}, 100%, 50%)'
        draw.rectangle(box, fill=None, outline=color, width=3)
        font = PIL.ImageFont.truetype('Arial.ttf', 14)
        draw.text(
            (box[0], max(box[1] - 20, 2)),
            str(round(100 * score))+" %",
            font=font,
            fill=(0, 255, 0, 255),
        )

    res_image.save(out_path)


if len(sys.argv) < 1:
    print('missing argument: image_path')
    print()
    print('usage: python3 face_finder.py image_path [output_path]')
    exit(1)

if __name__ == '__main__':
    net = Net()
    net.load_from_file()

    image_path = sys.argv[1]
    initial_image = PIL.Image.open(image_path)

    n_scales = 15
    scale_factor = 1 / auto_scale(n_scales, *initial_image.size)
    gray_image = initial_image.copy().convert('L')

    crop_w, crop_h = 36, 36
    stride = 15

    boxes, scores = pyramid_sliding_window_detection(
        net, gray_image, scale_factor, crop_w, crop_h, stride, minScore=0.9)

    print('Found', len(boxes), 'faces')
    draw_boxes_on_image(initial_image, boxes, scores)
