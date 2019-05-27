import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from DatasetBuilder import DatasetBuilder
import pydot


tf.enable_eager_execution()
tf.VERSION


PATCH_DIMENSION = 256
FEATURES = 9
encoded_bands = ['Coastal', 'Blue', 'Green', 'Yellow', 'Red', 'Red-edge', 'NIR1', 'NIR2']
source_Files = ['./training/tf_demo_image_classif_mali_chip_AA800_v2-00000.gz']
datasetBuilder = DatasetBuilder(PATCH_DIMENSION, encoded_bands)
image_batch, mask_batch = datasetBuilder.create_train_batch(image_paths=source_Files[0], mask_paths="./training/labels/*_mask.png")

# batch size is always omitted
def model():
	img_inputs = keras.Input(shape=(PATCH_DIMENSION, PATCH_DIMENSION, FEATURES), name='input_img')
	en_layer1_conv1 = layers.Conv2D(16, (3, 3), activation='relu', padding='SAME')(img_inputs)
	
	en_layer2_pool = layers.MaxPooling2D(2)(en_layer1_conv1)
	en_layer2_conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='SAME')(en_layer2_pool)
	
	en_layer3_pool = layers.MaxPooling2D(2)(en_layer2_conv1)
	en_layer3_conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='SAME')(en_layer3_pool)
	
	en_layer4_pool = layers.MaxPooling2D(2)(en_layer3_conv1)
	en_layer4_conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='SAME')(en_layer4_pool)
	
	en_layer5_pool = layers.MaxPooling2D(2)(en_layer4_conv1)
	en_layer5_conv1 = layers.Conv2D(128, (3, 3), activation='relu', padding='SAME')(en_layer5_pool)
	en_layer5_deconv = layers.Conv2DTranspose(128, 3, strides=(2, 2), activation='relu', padding='SAME')(en_layer5_conv1)

	# # Layer 4 deconvolution
	de_layer4_concat = layers.concatenate([en_layer5_deconv,en_layer4_conv1], axis=3)
	de_layer4_conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='SAME')(de_layer4_concat)
	de_layer4_deconv = layers.Conv2DTranspose(32, 3, strides=(2, 2), activation='relu', padding='SAME')(de_layer4_conv1)

	# # Layer 3 deconvolution
	de_layer3_concat = layers.concatenate([de_layer4_deconv,en_layer3_conv1], axis=3)
	de_layer3_conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='SAME')(de_layer3_concat)
	de_layer3_deconv = layers.Conv2DTranspose(32, 3, strides=(2, 2), activation='relu', padding='SAME')(de_layer3_conv1)

	# # Layer 2 deconvolution
	de_layer2_concat = layers.concatenate([de_layer3_deconv,en_layer2_conv1], axis=3)
	de_layer2_conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='SAME')(de_layer2_concat)
	de_layer2_deconv = layers.Conv2DTranspose(16, 3, strides=(2, 2), activation='relu', padding='SAME')(de_layer2_conv1)

	# # concat
	de_layer1_concat = layers.concatenate([de_layer2_deconv,en_layer1_conv1], axis=3, name='concat_l1')
	outputs = layers.Conv2D(1, (3, 3), activation='relu', padding='SAME')(de_layer1_concat)

	return img_inputs, outputs

img_inputs, outputs = model()
model = keras.Model(inputs=img_inputs, outputs=outputs)

model.summary()
# keras.utils.plot_model(model, 'my_first_model_with_shape_info.png', show_shapes=True)

# TRAIN model
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])

model.fit(image_batch, mask_batch, steps_per_epoch=10, epochs=1)

# EVALUATE model
test_loss, test_acc = model.evaluate(image_batch, mask_batch, steps=1)
print(test_acc)

# PREDICT data
def showPrediction(datasetBuilder, img_batch, mask_batch, pred_batch, index):
	img = img_batch[index,:,:,:]
	datasetBuilder.show_image(img)

	# img = mask_batch[index,:,:,:]
	# datasetBuilder.show_image(img)

	prediction = pred_batch[index,:,:,:]
	datasetBuilder.show_image(prediction)

pred_image_batch = datasetBuilder.create_pred_batch(image_paths=source_Files[0])
print(pred_image_batch.shape)

predictions = model.predict(pred_image_batch, steps=1)
print(predictions.shape)

showPrediction(datasetBuilder, pred_image_batch, mask_batch, predictions, 0)
# showPrediction(datasetBuilder, pred_image_batch, mask_batch, predictions, 1)
# showPrediction(datasetBuilder, pred_image_batch, mask_batch, predictions, 2)