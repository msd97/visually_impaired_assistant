import cv2 
import time
import logging
import tensorflow as tf
import numpy as np
from keras.models import load_model
from DenseDepth.layers import BilinearUpSampling2D
from PIL import Image
from ssd_tf_hub import inference_hub, draw_boxes
from mono_est import depth_est

calc_depth = 1
edge = 1

logging.basicConfig(format="%(asctime)s // %(levelname)s : %(message)s", datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)

# Function to initialize video streamer in NVIDIA Jetson Nano
def gstreamer_pipeline(
    sensor_id=0,
    #sensor_mode=3,
    capture_width=3280,
    capture_height=2464,
    display_width=816,
    display_height=616,
    framerate=21/1,
    flip_method=2,
):
    return (
       "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=1"
        % (
            sensor_id,
            #sensor_mode,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

if edge:
    from pycoral.adapters import common
    from pycoral.adapters import detect
    from pycoral.utils.dataset import read_label_file
    from pycoral.utils import edgetpu
    m_path = 'tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite'
    interpreter = edgetpu.make_interpreter(m_path)
    interpreter.allocate_tensors()
    labels = read_label_file('coco_labels.txt')

else:
    m_path = 'ssd_mobilenet_v2_2'
    model = tf.saved_model.load(m_path)

if calc_depth:
    d_path = 'nyu.h5'
    custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

    d_model = load_model(d_path, custom_objects=custom_objects, compile=False)
    logging.info('Monocular Depth model successfully loaded')


logging.info('Detection model successfully loaded')

#cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER)
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('people_street.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_detection = cv2.VideoWriter('detection.avi', fourcc, 20.0, (854, 480))
out_map = cv2.VideoWriter('dmap.avi', fourcc, 20.0, (320, 240))

#ret,frame = cap.read()
#cv2.imshow('frame', frame)

logging.info('Video streamer initialized')

# used to record the time when we processed last frame
prev_frame_time = 0
 
# used to record the time at which we processed current frame
new_frame_time = 0

if not edge:
    labels = []
    with open('coco_labels.txt', 'r') as coco:
        classes = coco.readlines()
        for i, line in enumerate(classes):
            labels.append(line.strip())

  
while(True):
    
    # Capture the video frame
    # by frame
    ret, frame = cap.read()

    new_frame_time = time.time()

    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time

    if edge:
        image = Image.fromarray(frame)
        _, scale = common.set_resized_input(
            interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))
        start = time.perf_counter()
        interpreter.invoke()
        inference_time = time.perf_counter() - start
        objs = detect.get_objects(interpreter, 0.4, scale)
        logging.info('%.2f ms' % (inference_time * 1000))
        boxes = []
        class_names = []
        scores = []
        for obj in objs:
            print(labels.get(obj.id, obj.id))
            print('  id:    ', obj.id)
            print('  score: ', obj.score)
            print('  bbox:  ', obj.bbox)
            class_names.append(obj.id)
            scores.append(obj.score)
            boxes.append(obj.bbox)
        
        frame, raw_boxes, scaled_boxes = draw_boxes(frame, boxes, class_names, scores, labels)

    else:
        detection, raw_bbox, scaled_bbox = inference_hub(frame.copy(), model, labels)

    if calc_depth:
        d_map_show, d_map = depth_est(d_model, frame.copy())
        for i, box in enumerate(raw_bbox):
            H= d_map.shape[0]
            W = d_map.shape[1]
            ymin, xmin, ymax, xmax = tuple(box)
            xmin = int(W * xmin)
            ymin = int(H * ymin)
            xmax = int(W * xmax)
            ymax = int(H * ymax)
            #avg_depth = np.mean(d_map[ymin:ymax, xmin:xmax])
            avg_depth = d_map[int((ymax-ymin)/2), int((xmax-xmin)/2), 0]
            ymin, xmin, ymax, xmax = tuple(scaled_bbox[i])
            if 1/avg_depth > 3:
                cv2.putText(detection, 'TOO CLOSE', (xmin, ymax-11), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        cv2.imshow('depth', d_map_show)
    #detection = cv2.cvtColor(detection, cv2.COLOR_RGB2BGR)
    # Display the resulting frame
    cv2.putText(detection, 'FPS: {:.2f}'.format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    cv2.imshow('frame', detection)
    out_detection.write(detection)
    out_map.write(d_map_show)
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
cap.release()
out_detection.release()
out_map.release()
# Destroy all the windows
cv2.destroyAllWindows()