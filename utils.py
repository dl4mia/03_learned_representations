import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from PIL import Image, ImageSequence


def get_colormap():
    colors = [
        [0, 0, 0, 0.5],
        [0.45, 0.92, 0.40, 1.],
        [0.99, 0.45, 0.24, 1.],
        [0, 0.65, 0.97, 1.],
        [0.95, 00.95, 0.25, 1.],
        [0.84, 0.5, 0.97, 1.],
        [0.58, 0.16, 0.81, 1.],
    ]
    cm = mpl.colors.ListedColormap(colors, "dl4mia")

    return cm, colors


def tiff_force_8bit(image):
    array = np.array(image)
    arr_min = array.min()
    arr_max = array.max()
    if arr_max > 255:
        normalized = (array.astype(np.uint16) - arr_min) * 255.0 / (arr_max - arr_min)
        image = normalized.astype(np.uint8)
    else:
        image = array.astype(np.uint8)

    return image


def get_images_from_tiff(tiff_file, to_rgb=False):
    imgs = []
    for page in ImageSequence.Iterator(Image.open(tiff_file)):
        img = tiff_force_8bit(page)
        if to_rgb:
            img = img[:, :, np.newaxis].repeat(3, axis=-1)
        imgs.append(img)

    return np.stack(imgs)


def plot_data_sample(image, mask, cmap="Dark2"):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.5), layout="compressed")
    fig.canvas.toolbar_position = "right"
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False
    axes[0].imshow(image, cmap="grey")
    axes[0].set_title("Image")
    axes[1].imshow(mask, cmap=cmap, interpolation="none")
    axes[1].set_title("Label")
    axes[2].imshow(image, interpolation="none", cmap="grey")
    axes[2].imshow(mask, alpha=0.45, cmap=cmap, interpolation="none")
    axes[2].set_title("Overlay")
    for ax in axes:
        ax.set_yticks([])
        ax.set_aspect("equal", "box")

    plt.show()
