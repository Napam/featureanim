# https://stackoverflow.com/a/62781370/13091658

import numpy as np
import torch
from typing import Optional, Tuple


def get_training_images():
    with open("data/train-images-idx3-ubyte", "rb") as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), "big")
        # second 4 bytes is the number of images
        image_count = int.from_bytes(f.read(4), "big")
        # third 4 bytes is the row count
        row_count = int.from_bytes(f.read(4), "big")
        # fourth 4 bytes is the column count
        column_count = int.from_bytes(f.read(4), "big")
        # rest is the image pixel data, each pixel is stored as an unsigned byte
        # pixel values are 0 to 255
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8).reshape(
            (image_count, row_count, column_count)
        )
        return images


def get_training_labels():
    with open("data/train-labels-idx1-ubyte", "rb") as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), "big")
        # second 4 bytes is the number of labels
        label_count = int.from_bytes(f.read(4), "big")
        # rest is the label data, each label is stored as unsigned byte
        # label values are 0 to 9
        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
        return labels


def dump_img(index: int):
    from matplotlib import pyplot as plt

    images = get_training_images()
    plt.imsave(f"image.png", images[index])


def generator(
    batch_size: int = 512,
    range_: Optional[Tuple[int, int]] = None,
    device: Optional[torch.device] = None,
):
    imagess = get_training_images()
    labelss = get_training_labels()

    indices = np.arange(len(imagess))
    np.random.shuffle(indices)

    imagess = imagess[indices]
    labelss = labelss[indices]

    if range_ is None:
        range_ = (0, len(imagess))

    imagess = torch.tensor(imagess, dtype=torch.float32, device=device)[range_[0] : range_[1]]
    labelss = torch.tensor(labelss, dtype=torch.long, device=device)[range_[0] : range_[1]]

    n_batches = len(imagess) // batch_size

    images = imagess[: n_batches * batch_size] / 255
    labels = labelss[: n_batches * batch_size] 

    imagess = images.view(n_batches, batch_size, -1, 28, 28)
    labelss = labels.view(n_batches, batch_size, -1)

    for images, labels in zip(imagess, labelss):
        yield images, labels


if __name__ == "__main__":
    gen = generator()
    for X, y in gen:
        exit(y)
    