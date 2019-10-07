import base64
import colorsys
import logging
import time

import cv2
import pika
import tensorflow as tf
from skimage.measure import find_contours

from FAPSDemonstrator.demonstrator import *

levels = (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL)
# logging.basicConfig(format='%(asctime)-15s [%(levelname)] [%(name)-12s] %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger('FAPS Image Processing Service')
logger.setLevel(logging.DEBUG)
DEMONSTRATOR_ENDPOINT = "cloud.faps.uni-erlangen.de"

# Directory to save logs and trained model
MODEL_DIR = "D:\PYTHON_Workspace\Mask_RCNN\FAPSDemonstrator\logs"

# Path to the trained weights
# You can download this file from the Releases page
LAST_WEIGHTS_PATH = "E:\WORKSPACE_JBK\GITHUB\Mask_RCNN\FAPSDemonstrator\logs/fapsdemonstrator20191006T0325\mask_rcnn_fapsdemonstrator_0006.h5"  # TODO: update this
CLASS_NAMES = ['BG', "peanuts", "m_and_m", "haribo", "faps"]

# Device to load the neural network on.
# Useful if you're training a model on the same
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0


class ImageProcessor:
    def __init__(self):
        config_class = FAPSDemonstratorConfig

        # Override the training configurations with a few
        # changes for inferencing.
        class ImageProcessorInferenceConfig(config_class):
            # Run detection on one image at a time
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        self.config = ImageProcessorInferenceConfig()
        self.config.display()

        # Create model in inference mode
        with tf.device(DEVICE):
            self.model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=self.config)

        # Load weights trained
        self.model.load_weights(LAST_WEIGHTS_PATH, by_name=True)

        self.class_names = CLASS_NAMES
        self.ground_colors = self.get_colors(len(self.class_names))

    def apply_mask(self, image, mask, color, alpha=0.5):
        """Apply the given mask to the image.
        """
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c],
                                      image[:, :, c])
        return image

    def draw_bbox(self, image, bboxes, masks, class_ids, class_names, scores, colors, show_label=True, show_mask=True):
        """
        boxes, masks, class_ids, class_names, scores,colors=colors
        bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
        """
        image_h, image_w, _ = image.shape

        for i, bbox in enumerate(bboxes):
            y1, x1, y2, x2 = bboxes[i]
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
                image = self.apply_mask(image, mask, bbox_color)

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

    def get_colors(self, num_classes):
        hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
        return colors

    def process_image(self, image_path=None):
        logger.info("Processing new images")
        if image_path is None:
            return []

        # since we have only gray images
        image = skimage.io.imread(str(image_path))
        # for the FAPS demonstrator we only have grey images
        frame = skimage.color.gray2rgb(image)

        start = time.time()
        results = self.model.detect([frame], verbose=1)
        r = results[0]
        end = time.time()
        logger.info("Inference time: {:.2f}s".format(end - start))

        boxes = r['rois']
        masks = r['masks']
        class_ids = r['class_ids']
        scores = r['scores']

        masked_image = self.draw_bbox(frame, boxes, masks, class_ids, self.class_names, scores, self.ground_colors)
        masked_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)

        return boxes, masks, class_ids, scores, masked_image


class Service:
    """This class is the class holdinfg the set of instruction that will be executed to the demonstrator."""

    def __init__(self, publish):
        self.imgs = []
        self.publish = publish
        self.connection = None
        self.channel = None
        self.exchange = None
        self.exchange_signal = None
        self.exchange_pub = None
        self.exchange_result_pub = None
        self.exchange_name = None
        self.exchange_signals_name = None
        self.exchange_pub_name = None
        self.exchange_result_pub_name = None
        self.queue = None
        self.queue_signal = None
        self.queue_pub = None
        self.queue_result_pub = None
        self.signal_queue = []
        self.processor = ImageProcessor()

    def incoming_picture_callback(self, ch, method, properties, data):
        logger.info("Incoming Image")
        if len(self.signal_queue) > 0:
            info_wrapper = self.signal_queue.pop(0)
            current_object = info_wrapper["object"]
            body = json.loads(data)
            buffer = np.asarray(body["value"]["data"]["data"])
            dir_path = os.path.dirname(os.path.realpath(__file__))
            if not os.path.exists(dir_path + '/input_images/'):
                os.makedirs(dir_path + '/input_images/')
            if not os.path.exists(dir_path + '/processed_images/'):
                os.makedirs(dir_path + '/processed_images/')
            file_path = dir_path + '/input_images/{}.bmp'.format(current_object)
            f = open(file_path, 'w+b')
            binary_format = bytearray(body["value"]["data"]["data"])
            f.write(binary_format)
            f.close()
            # Process the image. TODO: start an external process for it
            boxes, masks, class_ids, scores, masked_image = self.processor.process_image(file_path)

            # Save the processed image locally
            processed_file_path = dir_path + '/processed_images/{}.png'.format(current_object)
            cv2.imwrite(processed_file_path, masked_image)

            # Send the result back
            data = {
                "time": datetime.datetime.now().timestamp(),
                "start": True,
                "object": current_object,
                "info_wrapper": info_wrapper,
                "boxes": boxes.tolist(),
                "class_ids": class_ids.tolist(),
                "class_name": [CLASS_NAMES[c] for c in class_ids.tolist()],
                "scores": scores.tolist()
            }
            self.channel.basic_publish(
                exchange=self.exchange_pub_name,
                routing_key='',
                body=json.dumps(data))

            # Send the debug picture back
            # Display the resulting frame
            if self.publish is False:
                cv2.namedWindow(current_object, cv2.WINDOW_NORMAL)
                # cv2.resizeWindow('image', 192, 256)
                # cv2.moveWindow(image_path, 100, 100)
                cv2.imshow(current_object, masked_image)
                if cv2.waitKey(2) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    return
            else:
                bytes_read = open(processed_file_path, "rb").read()
                debug_data = {
                    "time": datetime.datetime.now().timestamp(),
                    "start": True,
                    "object": current_object,
                    "info_wrapper": info_wrapper,
                    "boxes": boxes.tolist(),
                    "class_ids": class_ids.tolist(),
                    "class_name": [CLASS_NAMES[c] for c in class_ids.tolist()],
                    "scores": scores.tolist(),
                    "picture": bytes_read.hex()
                }
                self.channel.basic_publish(
                    exchange=self.exchange_result_pub_name,
                    routing_key='',
                    body=json.dumps(debug_data))

    def incoming_processing_signal_callback(self, ch, method, properties, data):
        logger.info("Image Processing Signal")
        body = json.loads(data)
        if body["start"] == True:
            self.signal_queue.append(body)

    def connect_and_start(self, _url, _port, _user, _passwd, _exchange, _queue, _exchange_signal, _queue_signal,
                          _exchange_pub, _queue_pub, _exchange_result_pub, _queue_result_pub,
                          _callback_incoming_image, _callback_image_signal):
        """
            Connect the FAPSDemonstratorAPI to the demonstrator.
        :return true if the connect has been established or false otherwise.
        """
        try:
            self.connection = pika.BlockingConnection(pika.ConnectionParameters(
                port=_port,
                host=_url,
                credentials=pika.PlainCredentials(_user, _passwd))
            )
            self.channel = self.connection.channel()
            self.exchange_name = _exchange
            self.exchange =self.channel.exchange_declare(
                exchange=_exchange,
                passive=False,
                durable=False,
                exchange_type='fanout'
            )
            self.exchange_signals_name = _exchange_signal
            self.exchange_signal = self.channel.exchange_declare(
                exchange=_exchange_signal,
                passive=False,
                durable=False,
                exchange_type='fanout'
            )
            self.exchange_pub_name = _exchange_pub
            self.exchange_pub = self.channel.exchange_declare(
                exchange=_exchange_pub,
                passive=False,
                durable=False,
                exchange_type='fanout'
            )
            self.exchange_result_pub_name = _exchange_result_pub
            self.exchange_result_pub = self.channel.exchange_declare(
                exchange=_exchange_result_pub,
                passive=False,
                durable=False,
                exchange_type='fanout'
            )

            self.queue = self.channel.queue_declare(
                queue=_queue,
                durable=False,
                exclusive=False,
                auto_delete=True
            ).method.queue
            self.queue_signal = self.channel.queue_declare(
                queue=_queue_signal,
                durable=False,
                exclusive=False,
                auto_delete=True
            ).method.queue
            self.queue_pub = self.channel.queue_declare(
                queue=_queue_pub,
                durable=False,
                exclusive=False,
                auto_delete=True,
            ).method.queue
            self.queue_result_pub = self.channel.queue_declare(
                queue=_queue_result_pub,
                durable=False,
                exclusive=False,
                auto_delete=True,
            ).method.queue

            self.channel.queue_bind(exchange=_exchange, queue=self.queue, routing_key='')
            self.channel.queue_bind(exchange=_exchange_signal, queue=self.queue_signal, routing_key='')
            self.channel.queue_bind(exchange=_exchange_pub, queue=self.queue_pub, routing_key='')

            # bind the call back to the demonstrator FAPSDemonstratorAPI and start listening
            self.channel.basic_consume(on_message_callback=_callback_incoming_image, queue=self.queue, auto_ack=True)
            self.channel.basic_consume(on_message_callback=_callback_image_signal, queue=self.queue_signal, auto_ack=True)
            try:
                self.channel.start_consuming()
            except KeyboardInterrupt:
                self.connection.close()
                exit(0)

            return self.connection, self.channel

        except Exception as e:
            logger.error(e)
            if not (self.channel is None):
                self.channel.close()
                self.channel = None
            if not (self.connection is None):
                self.connection.close()
                self.connection = None
            return None, None


if __name__ == '__main__':
    logger.info('Demonstrator Image Processing Service using pika version: %s' % pika.__version__)
    service = Service(True)

    service.connect_and_start(
        _url=DEMONSTRATOR_ENDPOINT,
        _port=5672,
        _user='esys',
        _passwd='esys',
        _exchange="FAPS_DEMONSTRATOR_ImageProcessing_CameraPictures",
        _queue="FAPS_DEMONSTRATOR_ImageProcessing_CameraPictures",
        _exchange_signal="FAPS_DEMONSTRATOR_ImageProcessing_ProcessingSignals",
        _queue_signal="FAPS_DEMONSTRATOR_ImageProcessing_ProcessingSignals",
        _exchange_pub="FAPS_DEMONSTRATOR_ImageProcessing_ProcessingResults",
        _queue_pub="FAPS_DEMONSTRATOR_ImageProcessing_ProcessingResults",
        _exchange_result_pub="FAPS_DEMONSTRATOR_ImageProcessing_ProcessingResults_Debug",
        _queue_result_pub="FAPS_DEMONSTRATOR_ImageProcessing_ProcessingResults_Debug",
        _callback_incoming_image=service.incoming_picture_callback,
        _callback_image_signal=service.incoming_processing_signal_callback
    )
