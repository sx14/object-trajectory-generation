import random
import matplotlib.pyplot as plt


def get_colors(n_color=100):
    """ Generate colors randomly """
    colors = []
    for i in range(n_color):
        r = random.randint(0, 255) / 255.0
        g = random.randint(0, 255) / 255.0
        b = random.randint(0, 255) / 255.0
        colors.append([r, g, b])
    return colors


def show_boxes(im, dets, cls, colors):
    """Draw detected bounding boxes."""
    if isinstance(im, str):
        im = plt.imread(im)
    plt.imshow(im, aspect='equal')

    for i in range(0, len(dets)):

        bbox = dets[i]

        if bbox is None:
            continue

        color = colors[i]
        rect = plt.Rectangle((bbox[0], bbox[1]),
                             bbox[2]-bbox[0],
                             bbox[3]-bbox[1], fill=False,
                             edgecolor=color, linewidth=3.5)
        plt.gca().add_patch(rect)
        plt.text(bbox[0], bbox[1] - 2, cls[i],
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    plt.axis('off')
    plt.tight_layout()
    plt.show()

