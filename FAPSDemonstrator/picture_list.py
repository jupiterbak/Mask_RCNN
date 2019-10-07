import colorsys
import os
import sys

import cv2
import numpy as np
import time
from skimage.measure import find_contours

import mrcnn.model as modellib
from mrcnn import utils
from mrcnn.config import Config
from pathlib import Path
from FAPSDemonstrator.demonstrator import *

# Directory to save logs and trained model
MODEL_DIR = "D:\PYTHON_Workspace\Mask_RCNN\FAPSDemonstrator\logs"

# Path to Ballon trained weights
# You can download this file from the Releases page
# https://github.com/matterport/Mask_RCNN/releases
LAST_WEIGHTS_PATH = "E:\WORKSPACE_JBK\GITHUB\Mask_RCNN\FAPSDemonstrator\logs/fapsdemonstrator20191006T0325\mask_rcnn_fapsdemonstrator_0006.h5"  # TODO: update this

config = FAPSDemonstratorConfig


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image


def draw_bbox(image, bboxes, masks, class_ids, class_names, scores, colors, show_label=True, show_mask=True):
    """
    boxes, masks, class_ids, class_names, scores,colors=colors
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    """
    image_h, image_w, _ = image.shape

    for i, bbox in enumerate(bboxes):
        y1, x1, y2, x2 = boxes[i]
        coor = np.array([x1, y1, x2, y2], dtype=np.int32)
        fontScale = 0.5
        score = scores[i]
        class_ind = int(class_ids[i])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % (class_names[class_ind], score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
            cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)  # filled

            cv2.putText(image, bbox_mess, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            image = apply_mask(image, mask, bbox_color)

            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            pts = np.array(contours[0], np.int32)
            pts = pts.reshape((-1, 1, 2))
            # image = cv2.polylines(image, [pts], True, bbox_color)

    return image


def get_colors(num_classes):
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    return colors


config = FAPSDemonstratorConfig
# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained
model.load_weights(LAST_WEIGHTS_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', "peanuts", "m_and_m", "haribo", "faps"]


fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
video_writer = cv2.VideoWriter('picture_list_output.avi', fourcc, 5.0, (1024, 768))
ground_colors = get_colors(len(class_names))

# get all images
PICTURE_FOLDER = "E:\WORKSPACE_JBK\Temp\Image_Labelling"
# Set path to the images
calib_imgs_path = Path(PICTURE_FOLDER)
img_list = []
calib_fnms = calib_imgs_path.glob('*.bmp')

for idx, fn in enumerate(calib_fnms):
    print(idx, '', end='')
    # since we have only gray images
    image = skimage.io.imread(str(calib_imgs_path.joinpath(fn)))
    # for the FAPS demonstrator we only have grey images
    frame = skimage.color.gray2rgb(image)
    # frame, _, _, _, _ = utils.resize_image(
    #     image,
    #     min_dim=config.IMAGE_MIN_DIM,
    #     max_dim=config.IMAGE_MAX_DIM,
    #     mode=config.IMAGE_RESIZE_MODE)

    start = time.time()
    results = model.detect([frame], verbose=1)
    r = results[0]
    end = time.time()
    print("Inference time: {:.2f}s".format(end - start))

    boxes = r['rois']
    masks = r['masks']
    class_ids = r['class_ids']
    scores = r['scores']

    masked_image = draw_bbox(frame, boxes, masks, class_ids, class_names, scores, ground_colors)
    # masked_image = visualize.display_instances_images(frame, boxes, masks, class_ids, class_names, scores,
    #                                                  colors=colors)
    # masked_image = visualize.get_masked_image(frame, boxes, masks, class_ids, class_names, scores, colors=colors)

    masked_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)

    # Write the video
    video_writer.write(masked_image)

    # Display the resulting frame
    cv2.imshow('', masked_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
video_writer.release()
cv2.destroyAllWindows()
