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

ROOT_DIR = os.path.abspath("./")


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
        y1, x1, y2, x2 = bbox[i]
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


# Demonstrator Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['M_and_M']

# Initiate SIFT detector
orb = cv2.ORB_create()

# Train class features
# for class_name in class_names:
#     template = cv2.imread('demonstrator_classes/{}.png'.format(class_name), 0)
#     w, h = template.shape[::-1]

template = cv2.imread('demonstrator_classes/M_and_M.png', 0)
w, h = template.shape[::-1]
kp1, des1 = orb.detectAndCompute(template, None)


stream = cv2.VideoCapture("assets/Test_Objects.mp4")
stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
video_writer = cv2.VideoWriter('output.avi', fourcc, 5.0, (1280, 720))
ground_colors = get_colors(len(class_names))

while True:
    # Capture frame-by-frame
    grabbed, frame = stream.read()
    if not grabbed:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Run detection
    start = time.time()
    kp2, des2 = orb.detectAndCompute(frame, None)
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1, des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    end = time.time()
    print("Inference time: {:.2f}s".format(end - start))

    # Draw a rectangle around the matched region.
    # Draw first 10 matches.
    img3 = cv2.drawMatches(template, kp1, frame, kp2, matches[:10], None,
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Write the video
    # video_writer.write(frame)

    # Display the resulting frame
    cv2.imshow('', img3)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
video_writer.release()
stream.release()
cv2.destroyAllWindows()
