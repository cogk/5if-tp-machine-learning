import PIL
import math
import dataclasses
import torchvision.transforms as transforms


transform = transforms.Compose(
    [transforms.Grayscale(),
     transforms.ToTensor(),
     transforms.Normalize(mean=(0,), std=(1,))])


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
