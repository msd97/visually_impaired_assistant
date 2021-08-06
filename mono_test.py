import cv2
import glob
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from keras.models import load_model
from DenseDepth.layers import BilinearUpSampling2D
from DenseDepth.utils import predict, load_images, display_images, to_multichannel

m_path = 'nyu.h5'
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

model = load_model(m_path, custom_objects=custom_objects, compile=False)

im_paths = glob.glob('./cal_left/*.png')[:20]
#test_im = cv2.imread(im_paths[0])
test_im = Image.open(im_paths[9])
in_im= np.asarray(test_im, dtype=float)
print(np.ptp(test_im, axis=0))
#in_im = np.array(test_im, dtype=float)
in_im = np.clip(in_im / 255, 0, 1)
in_im = np.expand_dims(in_im, axis=0)
print(in_im.shape)
print(in_im.dtype)
d_map = predict(model, in_im, batch_size=1)
d_map = np.squeeze(d_map, axis=0)
print((1/d_map)[:5,:5])
d_map = d_map / np.max(d_map)
d_map = d_map * 255
d_map = cv2.applyColorMap(d_map.astype(np.uint8), cv2.COLORMAP_PLASMA)
cv2.imshow('depth', d_map)

images = load_images(im_paths[:9])
print('\nLoaded ({0}) images of size {1}.'.format(images.shape[0], images.shape[1:]))

maps = predict(model,images, batch_size=1)

print(maps.shape)
print(maps.dtype)

viz = display_images(maps.copy(), images.copy())
fig, ax = plt.subplots(2,1,figsize=(10,10))
ax[0].imshow(viz)
ax[1].imshow(d_map)
#plt.savefig('test.png')
plt.show()