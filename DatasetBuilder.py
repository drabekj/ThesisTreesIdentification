import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob

class DatasetBuilder(object):
	"""docstring for DatasetBuilder"""
	def __init__(self, patch_dimensions, encoded_bands):
		super(DatasetBuilder, self).__init__()
		self.PATCH_DIMENSIONS = patch_dimensions
		self.ENCODED_BANDS = encoded_bands

	def generate_images(self, out_file_prefix, source_path, count, ndvi_flag=False):
		"""
		Generates and saves images (patches) from TFRecord encoded file.
		This can be used for manual mask creation.

		Args:
			out_file_prefix: (string)  Prefix of the generated files name,
				number will be appended.
			source_path:  (string)  TFRecord file path, which encodes Earth
				Engine exported image patches.
			count:  (int)  Count of images (patches) to generate.
			ndvi_flag:  (boolean)  Flag if in addition to RGB images, NDVI
				version should be generated as well.
		"""
		dataset = self.__create_img_dataset(source_path)

		for index, element in enumerate(dataset.take(count)):
			if ndvi_flag:
				band_NDVI = element[:,:,8]
				self.save_image(band_NDVI, out_file_prefix + "_" + str(index) + "_NDVI.png")

			if len(element.shape) > 2:
				element = self.__reduce_to_rgb(element)
			self.save_image(element, out_file_prefix + "_" + str(index) + ".png")

	def create_train_batch(self, image_paths, mask_paths):
		image_ds, mask_ds = self.create_train_dataset(image_paths, mask_paths)

		BATCH_SIZE = 2

		image_ds = image_ds.batch(BATCH_SIZE)
		mask_ds = mask_ds.batch(BATCH_SIZE)
		image_ds = image_ds.repeat()
		mask_ds = mask_ds.repeat()

		image_ds = image_ds.make_one_shot_iterator().get_next()
		mask_ds = mask_ds.make_one_shot_iterator().get_next()

		return image_ds, mask_ds

	def create_train_dataset(self, image_paths, mask_paths):
		"""
		Create datasets, both for images and masks.

		Args:
			image_paths:  (string)  TFRecord file path, which encodes Earth
				Engine exported image patches.
			mask_paths:  (string)  Mask path wildcarded.
				eg. ["./training/labels/*.png"]

		Images:
			- shape (PATCH_DIMENSIONS,PATCH_DIMENSIONS,features=9)
			- pixel values interval (0;1)
		Masks:
			- shape (256,256,1)
			- pixel values discrete [0,1]
		"""
		# get all mask paths as list
		mask_paths = glob.glob(mask_paths)
		mask_count = len(mask_paths)

		print("DATASET: There are {} mask files => there will be {} unique images in both images and masks.".format(mask_count, mask_count))

		image_ds = self.__create_img_dataset(image_paths).take(mask_count)
		mask_ds = self.__create_mask_dataset(mask_paths)

		return image_ds, mask_ds

	def create_pred_batch(self, image_paths):
		"""
		Create datasets, both for images and masks.

		Args:
			image_paths:  (string)  TFRecord file path, which encodes Earth
				Engine exported image patches.

		Images:
			- shape (PATCH_DIMENSIONS,PATCH_DIMENSIONS,features=9)
			- pixel values interval (0;1)
		"""
		# get all mask paths as list
		image_ds = self.__create_img_dataset(image_paths)

		image_ds = image_ds.batch(5)
		image_ds = image_ds.repeat()

		return image_ds.make_one_shot_iterator().get_next()

	def __create_mask_dataset(self, source_paths):
		"""
		Creates dataset for mask files (PNG) from provided path.

		Args:
			source_paths:  (list(string))  Mask files paths as list.
				eg. ["./training/labels/1_mask.png"]

		Returns:

		"""
		def load_and_preprocess_mask(self, path):
			image = tf.read_file(path)

			image = tf.image.decode_png(image, channels=3)
			image = tf.image.resize_images(image, [256, 256])

			print(image.shape)

			# reshape mask from 3 channels to 1 (drop 2)
			image, _ = tf.split(image, [1, 2], 2)
			image /= 255.0  # normalize to [0,1] range
			image = tf.cast(image, tf.int32)

			return image

		paths_ds = tf.data.Dataset.from_tensor_slices(source_paths)
		mask_ds = paths_ds.map(lambda path: load_and_preprocess_mask(self, path))

		return mask_ds

	def __create_img_dataset(self, source_path):
		"""
		Creates dataset from the TFRecord files exported from an image.
		Reshapes the data into (PATCH_DIMENSIONS, PATCH_DIMENSIONS, FEATURES).
		Creates and adds NDVI feature.

		Args:
			source_path:  (string)  TFRecord file path, which encodes Earth
				Engine exported image patches.

		Returns:
			DatasetAdapter of shape (PATCH_DIMENSIONS, PATCH_DIMENSIONS, selected_features)
			eg. <DatasetV1Adapter shapes: (256, 256, 9), types: tf.float32>
		"""
		def createFeaturesDict(self):
			# Note that the tensors are in the shape of a patch, one patch for each band.
			columns = [
			  tf.FixedLenFeature(shape=[self.PATCH_DIMENSIONS, self.PATCH_DIMENSIONS], dtype=tf.float32) for k in self.ENCODED_BANDS
			]

			featuresDict = dict(zip(self.ENCODED_BANDS, columns))
			return featuresDict

		def parse_image(self, example_proto, featuresDict):
			parsed_features = tf.parse_single_example(example_proto, featuresDict)

			# create NDVI feature (assuming NIR1 and Red are one of ENCODED_BANDS)
			ndvi_feature = tf.math.divide(tf.math.subtract(parsed_features['NIR1'], parsed_features['Red']), tf.math.add(parsed_features['NIR1'], parsed_features['Red']))

			features = [
			  parsed_features[k] for k in self.ENCODED_BANDS
			]
			features.append(ndvi_feature)

			return tf.stack(features, axis=2)

		image_ds = tf.data.TFRecordDataset(source_path, compression_type='GZIP')
		featuresDict = createFeaturesDict(self)

		image_ds = image_ds.map(lambda x: parse_image(self, x, featuresDict))

		return image_ds

	def __reduce_to_rgb(self, img):
		"""
		Expects data where:
			channel 4 = Red
			channel 2 = Green
			channel 1 = Blue
		"""
		try:
			band_R = img[:,:,4]
			band_G = img[:,:,2]
			band_B = img[:,:,1]
		except:
			band_R = img[:,:,0]
			band_G = img[:,:,1]
			band_B = img[:,:,2]

		return np.stack((band_R, band_G, band_B), axis=2)

	def save_image(self, img, file_name):
		"""
		Save image to file. Img has to be prvided as matrix with dimensions:
			(self.PATCH_DIMENSIONS, self.PATCH_DIMENSIONS, 3) or
			(self.PATCH_DIMENSIONS, self.PATCH_DIMENSIONS, 1)
		"""
		sizes = np.shape(img)
		height = float(self.PATCH_DIMENSIONS)
		width = float(self.PATCH_DIMENSIONS)
		 
		fig = plt.figure()
		fig.set_size_inches(width/height, 1, forward=False)
		ax = plt.Axes(fig, [0., 0., 1., 1.])
		ax.set_axis_off()
		fig.add_axes(ax)

		ax.imshow(img)
		plt.savefig(file_name, dpi=height) 
		plt.close()

	def show_image(self, img):
		"""
		Show image in interactive window.
		"""
		if len(img.shape) > 2 and img.shape[2] > 1:
			img = self.__reduce_to_rgb(img)
			plt.imshow(img)
		else:
			plt.imshow(img[:,:,0])

		plt.show()




# tf.enable_eager_execution()
# tf.VERSION

# encoded_bands = ['Coastal', 'Blue', 'Green', 'Yellow', 'Red', 'Red-edge', 'NIR1', 'NIR2']
# source_Files = ['./training/tf_demo_image_classif_mali_chip_AA800_v2-00000.gz']
# datasetBuilder = DatasetBuilder(256, encoded_bands)

# # datasetBuilder.generate_images("generated_file", source_Files[0], count=10, ndvi_flag=True)
# image_batch, mask_batch = datasetBuilder.create_train_batch(image_paths=source_Files[0], mask_paths="./training/labels/*_mask.png")

# # print(image_batch.shape, mask_batch.shape)

# for batch in mask_batch:
# 	print(batch.shape)
# 	datasetBuilder.show_image(batch[0])