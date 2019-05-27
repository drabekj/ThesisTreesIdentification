import tensorflow as tf
import numpy as np
from DatasetBuilder import DatasetBuilder

# tf.enable_eager_execution()
# tf.VERSION

np.random.seed(678)
tf.set_random_seed(5678)

def tf_relu(x): return tf.nn.relu(x)
def d_tf_relu(s): return tf.cast(tf.greater(s,0),dtype=tf.float32)
def tf_softmax(x): return tf.nn.softmax(x)
def np_sigmoid(x): 1/(1 + np.exp(-1 *x))

# --- make class ---
class ConlayerLeft():
	def __init__(self,ker,in_c,out_c):
		self.w = tf.Variable(tf.random_normal([ker,ker,in_c,out_c],stddev=0.05))

	def feedforward(self,input,stride=1,dilate=1):
		self.input  = input
		self.layer  = tf.nn.conv2d(input,self.w,strides = [1,stride,stride,1],padding='SAME')
		self.layerA = tf_relu(self.layer)
		return self.layerA

class ConlayerRight():
	def __init__(self,ker,in_c,out_c):
		self.w = tf.Variable(tf.random_normal([ker,ker,in_c,out_c],stddev=0.05))

	def feedforward(self,input,stride=1,dilate=1,output=1):
		self.input  = input

		current_shape_size = input.shape

		self.layer = tf.nn.conv2d_transpose(input,self.w,
		output_shape=[batch_size] + [int(current_shape_size[1].value*2),int(current_shape_size[2].value*2),int(current_shape_size[3].value/2)],strides=[1,2,2,1],padding='SAME')
		self.layerA = tf_relu(self.layer)
		return self.layerA

# --- hyper ---
num_epoch = 100
init_lr = 0.0001
batch_size = 1

# --- make layer ---
# left
l1_1 = ConlayerLeft(3,9,3)
l1_2 = ConlayerLeft(3,3,3)
l1_3 = ConlayerLeft(3,3,3)

l2_1 = ConlayerLeft(3,3,6)
l2_2 = ConlayerLeft(3,6,6)
l2_3 = ConlayerLeft(3,6,6)

l3_1 = ConlayerLeft(3,6,12)
l3_2 = ConlayerLeft(3,12,12)
l3_3 = ConlayerLeft(3,12,12)

l4_1 = ConlayerLeft(3,12,24)
l4_2 = ConlayerLeft(3,24,24)
l4_3 = ConlayerLeft(3,24,24)

l5_1 = ConlayerLeft(3,24,48)
l5_2 = ConlayerLeft(3,48,48)
l5_3 = ConlayerLeft(3,48,24)

# right
l6_1 = ConlayerRight(3,24,48)
l6_2 = ConlayerLeft(3,24,24)
l6_3 = ConlayerLeft(3,24,12)

l7_1 = ConlayerRight(3,12,24)
l7_2 = ConlayerLeft(3,12,12)
l7_3 = ConlayerLeft(3,12,6)

l8_1 = ConlayerRight(3,6,12)
l8_2 = ConlayerLeft(3,6,6)
l8_3 = ConlayerLeft(3,6,3)

l9_1 = ConlayerRight(3,3,6)
l9_2 = ConlayerLeft(3,3,3)
l9_3 = ConlayerLeft(3,3,3)

l10_final = ConlayerLeft(3,3,3)

# ---- make graph ----
# x = tf.placeholder(shape=[None,256,256,1],dtype=tf.float32)
# y = tf.placeholder(shape=[None,256,256,1],dtype=tf.float32)
encoded_bands = ['Coastal', 'Blue', 'Green', 'Yellow', 'Red', 'Red-edge', 'NIR1', 'NIR2']
source_Files = ['tf_demo_image_classif_mali_chip_AA800_v2-00000.gz']
datasetBuilder = DatasetBuilder(256, encoded_bands)
image_ds, mask_ds = datasetBuilder.create_dataset(image_paths=source_Files[0], mask_paths="./training/labels/*_mask.png")
x = image_ds
y = mask_ds

layer1_1 = l1_1.feedforward(x)
layer1_2 = l1_2.feedforward(layer1_1)
layer1_3 = l1_3.feedforward(layer1_2)

layer2_Input = tf.nn.max_pool(layer1_3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer2_1 = l2_1.feedforward(layer2_Input)
layer2_2 = l2_2.feedforward(layer2_1)
layer2_3 = l2_3.feedforward(layer2_2)

layer3_Input = tf.nn.max_pool(layer2_3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer3_1 = l3_1.feedforward(layer3_Input)
layer3_2 = l3_2.feedforward(layer3_1)
layer3_3 = l3_3.feedforward(layer3_2)

layer4_Input = tf.nn.max_pool(layer3_3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer4_1 = l4_1.feedforward(layer4_Input)
layer4_2 = l4_2.feedforward(layer4_1)
layer4_3 = l4_3.feedforward(layer4_2)

layer5_Input = tf.nn.max_pool(layer4_3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer5_1 = l5_1.feedforward(layer5_Input)
layer5_2 = l5_2.feedforward(layer5_1)
layer5_3 = l5_3.feedforward(layer5_2)

layer6_Input = tf.concat([layer5_3,layer5_Input],axis=3)
layer6_1 = l6_1.feedforward(layer6_Input)
layer6_2 = l6_2.feedforward(layer6_1)
layer6_3 = l6_3.feedforward(layer6_2)

layer7_Input = tf.concat([layer6_3,layer4_Input],axis=3)
layer7_1 = l7_1.feedforward(layer7_Input)
layer7_2 = l7_2.feedforward(layer7_1)
layer7_3 = l7_3.feedforward(layer7_2)

layer8_Input = tf.concat([layer7_3,layer3_Input],axis=3)
layer8_1 = l8_1.feedforward(layer8_Input)
layer8_2 = l8_2.feedforward(layer8_1)
layer8_3 = l8_3.feedforward(layer8_2)

layer9_Input = tf.concat([layer8_3,layer2_Input],axis=3)
layer9_1 = l9_1.feedforward(layer9_Input)
layer9_2 = l9_2.feedforward(layer9_1)
layer9_3 = l9_3.feedforward(layer9_2)

layer10 = l10_final.feedforward(layer9_3)

cost = tf.reduce_mean(tf.square(layer10-y))
auto_train = tf.train.AdamOptimizer(learning_rate=init_lr).minimize(cost)


# --- start session ---
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	epochs = 5
	for i in range(epochs): 
		# sess.run(iterator.initializer)

		try:
		    # Go through the entire dataset
			while True:
				mask_ds = sess.run(mask_ds)
				print(mask_ds.shape)
				# print(mask_ds[0,101,139])
				# print(mask_ds[0,101,138])

				print('\n-----------------------')
		        
		except tf.errors.OutOfRangeError:
			print('End of Epoch.')

    # for iter in range(num_epoch):
    #     # train
    #     for current_batch_index in range(0,len(train_images),batch_size):
    #         # current_batch = train_images[current_batch_index:current_batch_index+batch_size,:,:,:]
    #         # current_label = train_labels[current_batch_index:current_batch_index+batch_size,:,:,:]
    #         # sess_results = sess.run([cost,auto_train],feed_dict={x:current_batch,y:current_label})
    #         sess_results = sess.run()
    #         print(' Iter: ', iter, " Cost:  %.32f"% sess_results[0],end='\r')
		print('\n-----------------------')
# -- end code --