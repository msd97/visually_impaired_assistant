import cv2
import glob
import numpy as np
from matplotlib import pyplot as plt
from keras.models import load_model
from DenseDepth.layers import BilinearUpSampling2D
from DenseDepth.utils import predict, load_images, display_images, to_multichannel

def depth_est(model, img):
    img = cv2.resize(img, (640,480))
    in_im = np.array(img, dtype=float)
    in_im = np.clip(in_im / 255, 0, 1)
    in_im = np.expand_dims(in_im, axis=0)
    d_map = predict(model, in_im, batch_size=1)
    d_map = np.squeeze(d_map, axis=0)
    d_map_show = d_map.copy()
    d_map_show = d_map_show / np.max(d_map_show)
    d_map_show = d_map_show * 255
    d_map_show = cv2.applyColorMap(d_map_show.astype(np.uint8), cv2.COLORMAP_PLASMA)

    return d_map_show, d_map

"""m_path = 'kitti.h5'
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}


model = load_model(m_path, custom_objects=custom_objects, compile=False)

im_paths = glob.glob('./cal_left/*.png')[:12]
print(im_paths)
images = load_images(im_paths)
print('\nLoaded ({0}) images of size {1}.'.format(images.shape[0], images.shape[1:]))

maps = predict(model,images, batch_size=1)

viz = display_images(maps.copy(), images.copy())
plt.figure(figsize=(10,5))
plt.imshow(viz)
plt.savefig('test.png')
plt.show()"""