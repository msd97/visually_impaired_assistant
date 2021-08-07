import logging
import cv2
import numpy as np
import tensorflow as tf

def draw_boxes(img, boxes, class_names, scores, true_labels, edge=False):

    boxes = np.array(boxes)
    max_boxes = 10
    min_score = 0.4
    raw_boxes = []
    scaled_boxes = []

    if edge:
        class_names = [x+1 for x in class_names]

    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score and int(class_names[i]) == 1:
            H= img.shape[0]
            W = img.shape[1]

            ymin, xmin, ymax, xmax = tuple(boxes[i])
            raw_boxes.append((ymin,xmin,ymax,xmax))

            xmin = int(W * xmin)
            ymin = int(H * ymin)
            xmax = int(W * xmax)
            ymax = int(H * ymax)

            scaled_boxes.append((ymin,xmin,ymax,xmax))

            label = true_labels[int(class_names[i]-1)]
            mark = label + " {:.2f}%".format(100*scores[i])
            logging.info('Obtained predictions for frame: {}'.format(mark))

            img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
            img = cv2.putText(img, mark, (xmin, ymin-11), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
    
    return img, raw_boxes, scaled_boxes


def inference_hub(img, model, true_labels):

    input_im = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    input_im = np.array(input_im)
    input_im = np.expand_dims(input_im, axis=0)
    input_im = tf.cast(input_im, tf.uint8)

    logging.info('Frame preprocessed, now performing inference')
    pred = model(input_im)

    boxes = np.squeeze(pred['detection_boxes'].numpy())
    labels = np.squeeze(pred['detection_classes'].numpy())
    scores = np.squeeze(pred['detection_scores'].numpy())
    marked, raw_boxes, scaled_boxes = draw_boxes(img, boxes, labels, scores, true_labels)

    return marked, raw_boxes, scaled_boxes
