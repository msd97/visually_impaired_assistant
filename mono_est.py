import cv2
import numpy as np
from DenseDepth.utils import predict

def depth_est(model, img):
    img = cv2.resize(img, (640,480))
    in_im = np.array(img, dtype=float)
    in_im = np.clip(in_im / 255, 0, 1)
    in_im = np.expand_dims(in_im, axis=0)
    d_map = predict(model, in_im, batch_size=1)
    d_map = np.squeeze(d_map, axis=0)
    d_map = d_map / np.max(d_map)
    d_map_show = d_map.copy()
    d_map_show = d_map_show * 255
    d_map_show = cv2.applyColorMap(d_map_show.astype(np.uint8), cv2.COLORMAP_PLASMA)

    return d_map_show, d_map