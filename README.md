# Visually Impaired People Smart Assistant for Outdoors Navigation

## 1. Installation

* Clone this repository
* Inside the repository's folder, clone the following repository: https://github.com/ialhashim/DenseDepth
* From the previous link, download nyu.h5 keras trained model and copy it to the visually impaired assistant folder
* Download trained SSD SavedModel directory from: https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2
* If using Edge TPU accelerator:
  * Change video_capture.py in line 13 from "edge = 0" to "edge = 1"
  * Download SSD trained model from: https://github.com/google-coral/test_data/raw/master/tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite
  * Copy TFLITE file to the visually impaired assistant folder

## 2. Run the assistant

The assistant runs by capturing a stream from a video file (original video taken from: https://www.youtube.com/watch?v=YzcawvDGe4Y). To try your own video file, change line 74 in *cv2.VideoCapture()* by passing as an argument to this function the path to your video. 

Run the assistant with video_capture.py script.

