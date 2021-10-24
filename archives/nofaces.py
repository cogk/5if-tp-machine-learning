import sys
import os
import re

import torch
import PIL

from ff_utils import generate_image_crops
from net import Net

if len(sys.argv) < 1:
    print('missing argument: image_path')
    exit(1)

if __name__ == '__main__':
    if not os.path.isdir('train_images_extra'):
        os.mkdir('train_images_extra')
    if not os.path.isdir('train_images_extra/0'):
        os.mkdir('train_images_extra/0')
    if not os.path.isdir('train_images_extra/1'):
        os.mkdir('train_images_extra/1')

    net = Net()
    net.load_from_file()

    image_path = sys.argv[1]
    initial_image = PIL.Image.open(image_path)
    img_id = re.sub(r'[^a-z0-9]+', '', image_path)

    results = []
    min_score = -999

    for d in generate_image_crops(initial_image):
        t = d.input
        cropped_image = d.img
        crop_id = d.id

        outputs = net(t[None, ...])

        score, predicted = torch.max(outputs.data, 1)
        score = score.item()
        predicted = predicted.item()

        # false positive
        if score > min_score and predicted == 1:
            p = f'./train_images_extra/0/{img_id}_{crop_id}.pgm'
            results.append((p, cropped_image, score))

    # only save "best" (worst) predictions
    results = sorted(results, reverse=True, key=lambda x: x[2])
    for (p, cropped_image, score) in results[:max(5, len(results)//5)]:
        print(p, score)
        cropped_image.save(p)
