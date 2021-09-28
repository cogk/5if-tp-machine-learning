import numpy as np
from itertools import islice
import PIL
import torch
from net import Net

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

    initial_image = load_image('./img/test.png')

    initial_size = tuple(initial_image.size)
    initial_w, initial_h = initial_size
    crop_w, crop_h = 36, 36
    stride_w, stride_h = crop_w // 2, crop_h // 2

    image = initial_image.copy()
    results = np.zeros((initial_w, initial_h))
    for z in range(10):
        W, H = image.size

        if W < crop_w or H < crop_h:
            print('break early')
            break

        for x in range(0, W - crop_w, stride_w):
            for y in range(0, H - crop_h, stride_h):
                left, top, right, bottom = x, y, x+crop_w, y+crop_h

                cropped_image = image.crop((left, top, right, bottom))
                cropped_image = transform(cropped_image)

                outputs = net(cropped_image[None, ...])
                _, predicted = torch.max(outputs.data, 1)
                predicted = predicted[0].item()

                if predicted > 0.5:
                    fx, fy = initial_w * left/W, initial_h * top/H
                    fw = initial_w * (right - left) / W
                    fh = initial_h * (bottom - top) / H
                    fx, fy, fw, fh = round(fx), round(fy), round(fw), round(fh)
                    results[fx:fx+fw, fy:fy+fh] += 1

        norm_results = results * 255 / np.max(results)
        norm_results = np.array(norm_results, dtype=np.uint8)

        res_image = PIL.Image.fromarray(norm_results)
        res_image = res_image.transpose(PIL.Image.TRANSPOSE)
        # res_image.save('./results/res' + str(z) + '.png')

        res_image2 = initial_image.copy()
        res_image2.putalpha(res_image)
        res_image2.save('./results/res.png')

        # resize source image
        newsize = (W/1.1, H/1.1)
        image.thumbnail(newsize, PIL.Image.ANTIALIAS)

        print('scaling')
        # break

    # cropped_image = torch.rand([1, 36, 36], dtype=torch.float)
