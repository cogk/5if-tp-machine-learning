import numpy as np
from itertools import islice
import PIL
import torch
from net import Net
import sys
import math
import random
import re

import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.Grayscale(),
     transforms.ToTensor(),
     transforms.Normalize(mean=(0,), std=(1,))])


def load_image(path):
    return PIL.Image.open(path)


def rolling_window(a, shape):  # rolling window for 2D array
    s = (a.shape[0] - shape[0] + 1,) + (a.shape[1] - shape[1] + 1,) + shape
    strides = a.strides + a.strides
    return np.lib.stride_tricks.as_strided(a, shape=s, strides=strides)


if __name__ == '__main__':
    net = Net()
    net.load_from_file()

    image_path = sys.argv[1] if len(sys.argv) > 1 else './img/test.png'
    initial_image = load_image(image_path)
    initial_image = initial_image

    img_id = re.sub(r'[^a-z0-9]+', '', image_path)

    initial_size = tuple(initial_image.size)
    initial_w, initial_h = initial_size
    crop_w, crop_h = 36, 36
    stride_w, stride_h = crop_w // 2, crop_h // 2

    N_SCALES = 15

    # We might want to detect portrait pictures (1 face per image)
    # so the scaling should go to ~36x36 (with correct aspect ratio)
    # We solve the following equation, for both width and height:
    # initial_w * SCALE_FACTOR ** N_SCALES = 36
    # <-> SCALE_FACTOR = (36 / initial_w) ** (1 / N_SCALES)
    # and select the biggest value
    SCALE_FACTOR = (36 / min(initial_w, initial_h)) ** (1 / (N_SCALES + 1))
    # SCALE_FACTOR = 0.95
    print('SCALE_FACTOR =', SCALE_FACTOR)

    image = initial_image.copy().convert('L')
    results = [0] * N_SCALES

    min_score = 0.175

    for z in range(N_SCALES):
        W, H = image.size

        if W < crop_w or H < crop_h:
            print('break early', W, H)
            break

        stride_w = int(min(
            (W / math.ceil(W/36)) // 2,
            crop_w // 2
        ))
        stride_h = int(min(
            (H / math.ceil(H/36)) // 2,
            crop_h // 2
        ))

        results[z] = np.zeros((initial_w, initial_h))

        for x in range(0, W - crop_w, stride_w):
            for y in range(0, H - crop_h, stride_h):
                left, top, right, bottom = x, y, x+crop_w, y+crop_h

                # id = f'{left}-{top}-{right}-{bottom}-{random.randint(0, 65535)}'

                cropped_image = image.crop((left, top, right, bottom))

                t = transform(cropped_image)
                outputs = net(t[None, ...])

                score, predicted = torch.max(outputs.data, 1)
                score = score.item()
                predicted = predicted.item()

                # cropped_image.save(
                #     f'./results/train/{predicted}/_{img_id}_{id}.pgm')

                # if predicted == 1:
                #     print(left, top, right, bottom, score, predicted)

                if score > min_score and predicted == 1:
                    fx, fy = initial_w * left/W, initial_h * top/H
                    fw = initial_w * (right - left) / W
                    fh = initial_h * (bottom - top) / H
                    fx, fy, fw, fh = round(fx), round(fy), round(fw), round(fh)

                    results[z][fx:fx+fw, fy:fy+fh] += score

        sum_results = sum(results)
        norm_results = np.copy(sum_results) * 255 / np.max(sum_results)
        # norm_results[norm_results < 0.1] = 0
        norm_results[norm_results > 0] = 255

        # binarize results
        # threshold = 0.001
        # norm_results[norm_results < threshold] = 0
        # norm_results[norm_results > threshold] = 1

        norm_results = np.array(norm_results, dtype=np.uint8)

        res_image = PIL.Image.fromarray(norm_results)
        res_image = res_image.transpose(PIL.Image.TRANSPOSE)
        # res_image.save('./results/res' + str(z) + '.png')

        res_image2 = initial_image.copy()
        res_image2.putalpha(res_image)
        res_image2.save('./results/res.png')

        # resize source image
        newsize = (initial_w * (SCALE_FACTOR ** (z + 1)),
                   initial_h * (SCALE_FACTOR ** (z + 1)))
        image.thumbnail(newsize, PIL.Image.ANTIALIAS)

        # if image.size[0] < 2*crop_w or image.size[1] < 2*crop_h:
        #     print('break')
        #     break
        print('scaling')
        # break

    # cropped_image = torch.rand([1, 36, 36], dtype=torch.float)
