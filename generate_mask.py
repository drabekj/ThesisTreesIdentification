import os
import numpy as np
import pandas as pd
import cv2
import shapely
import shapely.wkt
import matplotlib.pyplot as plt

def _get_xmax_ymin(image_id):
	xmax, ymin = gs[gs['ImageId'] == image_id].iloc[0, 1:].astype(float)
	return xmax, ymin

def get_scalers(height, width, x_max, y_min):
	"""
	:param height:
	:param width:
	:param x_max:
	:param y_min:
	:return: (xscaler, yscaler)
	"""
	w_ = width * (width / (width + 1))
	h_ = height * (height / (height + 1))
	return w_ / x_max, h_ / y_min

def polygons2mask_layer(height, width, polygons, image_id):
	"""
	:param height:
	:param width:
	:param polygons:
	:return:
	"""

	x_max, y_min = _get_xmax_ymin(image_id)
	x_scaler, y_scaler = get_scalers(height, width, x_max, y_min)

	polygons = shapely.affinity.scale(polygons, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))
	img_mask = np.zeros((height, width), np.uint8)

	if not polygons:
		return img_mask

	int_coords = lambda x: np.array(x).round().astype(np.int32)
	exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
	interiors = [int_coords(pi.coords) for poly in polygons for pi in poly.interiors]

	cv2.fillPoly(img_mask, exteriors, 1)
	cv2.fillPoly(img_mask, interiors, 0)
	return img_mask

def generate_mask(image_id, height, width, num_mask_channels, train):
	"""
	:param image_id:
	:param height:
	:param width:
	:param num_mask_channels: numbers of channels in the desired mask
	:param train: polygons with labels in the polygon format
	:return: mask corresponding to an image_id of the desired height and width with desired number of channels
	"""

	mask = np.zeros((num_mask_channels, height, width))

	for mask_channel in range(num_mask_channels):
		poly = train.loc[(train['ImageId'] == image_id)
						& (train['ClassType'] == mask_channel + 1), 'MultipolygonWKT'].values[0]
		polygons = shapely.wkt.loads(poly)
		mask[mask_channel, :, :] = polygons2mask_layer(height, width, polygons, image_id)
	return mask


data_path = './data'

gs = pd.read_csv(os.path.join(data_path, 'grid_sizes.csv'), names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
train_wkt = pd.read_csv(os.path.join(data_path, 'train_wkt_v4.csv'))
image_id = "6010_1_2"
num_mask_channels = 10

shapes = pd.read_csv(os.path.join(data_path, '3_shapes.csv'))
train_shapes = shapes[shapes['image_id'].isin(train_wkt['ImageId'].unique())]

min_train_height = train_shapes['height'].min()
min_train_width = train_shapes['width'].min()

print(min_train_height, min_train_width)

# image = extra_functions.read_image_16(image_id)
# _, height, width = image.shape
height, width = 3349, 3396

# generate_mask(image_id, height, width, num_mask_channels, train=train_wkt)[:, :min_train_height, :min_train_width]
img = generate_mask(image_id, height, width, num_mask_channels, train=train_wkt)
print(img.shape)
print(np.count_nonzero(img))


plt.imshow(img[7,:,:])
plt.show()