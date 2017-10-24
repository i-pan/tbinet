import keras 

from keras import backend as K 
from keras.engine import Input, Model 
from keras.layers import Conv3D, Conv3DTranspose, BatchNormalization
from keras.layers.merge import concatenate 
from keras.optimizers import Adam 

import numpy as np 
import nibabel as nib 
import datetime

from skimage.measure import block_reduce 

def double_conv_bn(x, num_fm, filter_size, activation="relu", padding="same",
	kernel_initializer="he_normal", kernel_regularizer=None):
	conv = Conv3D(num_fm, filter_size, activation=activation,
		padding=padding, kernel_initializer=kernel_initializer, 
		kernel_regularizer=kernel_regularizer)(x) 
	conv = BatchNormalization()(conv) 
	conv = Conv3D(num_fm, filter_size, activation=activation,
		padding=padding, kernel_initializer=kernel_initializer, 
		kernel_regularizer=kernel_regularizer)(conv)  
	conv = BatchNormalization()(conv) 
	return conv 

def local_wnet(inputs, filter_size, kernel_initializer, kernel_regularizer):
	# (32, 32, 32)
	l_conv1 = double_conv_bn(inputs, 32, filter_size, 
		kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
	l_down1 = Conv3D(64, (2,2,2), activation="relu", padding="same", 
		strides=(2,2,2), 
		kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(l_conv1)
	l_down1 = BatchNormalization()(l_down1) 
	# (16, 16, 16)
	l_conv2 = double_conv_bn(l_down1, 64, filter_size, 
		kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
	l_conv2 = keras.layers.add([l_conv2, l_down1])
	l_down2 = Conv3D(128, (2,2,2), activation="relu", padding="same", 
		strides=(2,2,2), 
		kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(l_conv2)
	l_down2 = BatchNormalization()(l_down2) 
	# (8, 8, 8)
	l_conv3 = double_conv_bn(l_down2, 128, filter_size, 
		kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
	l_conv3 = keras.layers.add([l_conv3, l_down2])
	l_up1 = Conv3DTranspose(128, (2,2,2), activation="relu", padding="same",
		strides=(2,2,2), 
		kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(l_conv3) 
	l_up1_0 = BatchNormalization()(l_up1)
	l_up1 = concatenate([l_conv2, l_up1_0], axis=4)
	# (16, 16, 16)
	return l_up1 

def global_wnet(inputs, filter_size, kernel_initializer, kernel_regularizer):
	# (32, 32, 32)
	g_conv1 = double_conv_bn(inputs, 16, filter_size, 
		kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
	g_down1 = Conv3D(32, (2,2,2), activation="relu", padding="same", 
		strides=(2,2,2), 
		kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(g_conv1)
	g_down1 = BatchNormalization()(g_down1) 
	# (16, 16, 16)
	g_conv2 = double_conv_bn(g_down1, 32, filter_size, 
		kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
	g_conv2 = keras.layers.add([g_conv2, g_down1])
	g_down2 = Conv3D(64, (2,2,2), activation="relu", padding="same", 
		strides=(2,2,2), 
		kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(g_conv2)
	g_down2 = BatchNormalization()(g_down2) 
	# (8, 8, 8)
	g_conv3 = double_conv_bn(g_down2, 64, filter_size, 
		kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
	g_conv3 = keras.layers.add([g_conv3, g_down2])
	g_down3 = Conv3D(128, (2,2,2), activation="relu", padding="same", 
		strides=(2,2,2),
		kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(g_conv3)
	# (4, 4, 4)
	g_conv4 = double_conv_bn(g_down3, 128, filter_size, 
		kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
	g_conv4 = keras.layers.add([g_conv4, g_down3])
	g_up1 = Conv3DTranspose(128, (2,2,2), activation="relu", padding="same",
		strides=(2,2,2), 
		kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(g_conv4) 
	g_up1_0 = BatchNormalization()(g_up1)
	g_up1 = concatenate([g_conv3, g_up1_0], axis=4)
	# (8, 8, 8)
	g_conv5 = double_conv_bn(g_up1, 64, filter_size,
		kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
	g_up2 = Conv3DTranspose(128, (2,2,2), activation="relu", padding="same",
		strides=(2,2,2), 
		kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(g_conv5) 
	g_up2_0 = BatchNormalization()(g_up2)
	g_up2 = concatenate([g_conv2, g_up2_0], axis=4)
	return g_up2

def load_wnet(input_shape=(32,32,32,1), filter_size=(3,3,3), n_classes=1, 
	initial_learning_rate=1e-3, kernel_initializer="he_normal", 
	kernel_regularizer=None): 
	l_inputs = Input(input_shape) ; g_inputs = Input(input_shape)
	l_out = local_wnet(l_inputs, filter_size, kernel_initializer, kernel_regularizer)
	g_out = global_wnet(g_inputs, filter_size, kernel_initializer, kernel_regularizer)
	combined = concatenate([l_out, g_out], axis=4)
	fc1 = Conv3D(256, (1,1,1), activation="relu", padding="same", 
		kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(combined)
	fc1 = BatchNormalization()(fc1)	
	fc2 = Conv3D(256, (1,1,1), activation="relu", padding="same", 
		kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(fc1)
	fc2 = BatchNormalization()(fc2)
	predictions = Conv3D(n_classes, (1,1,1), activation="sigmoid", padding="same", 
		kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(fc2)
	model = Model(inputs=[l_inputs, g_inputs], outputs=predictions) 
	model.compile(optimizer=Adam(initial_learning_rate), loss="binary_crossentropy")
	return model 

def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f ** 2) + K.sum(y_pred_f ** 2) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def load_pos_data(data_files, gt_files, num_cases, num_samples_per_case, 
	size=(32,32,32)):
	start_time = datetime.datetime.now() 
	print "Loading {} segments from {} positive cases ...".format(num_samples_per_case*num_cases, num_cases)
	X_case_list = [] 
	X2_case_list = [] 
	y_case_list = [] 
	indices = np.random.choice(range(len(data_files)), num_cases)
	pad_list = [(size[0],size[0]),(size[1],size[1]),(size[2],size[2])]
	for index in indices:
		data = nib.load(data_files[index]).get_data()
		gt = nib.load(gt_files[index]).get_data()
		data = np.pad(data, pad_list, "constant",
			constant_values=np.min(data))
		gt = np.pad(gt, pad_list, "constant")
		X,X2,y = get_X_y(data, gt, num_samples_per_case, size)
		X_case_list.append(X) ; X2_case_list.append(X2); y_case_list.append(y) 
	X_array = np.vstack(X_case_list)
	X2_array = np.vstack(X2_case_list)
	y_array = np.vstack(y_case_list) 
	print("DONE in {} !".format(datetime.datetime.now() - start_time))
	return np.expand_dims(X_array, axis=4), np.expand_dims(X2_array, axis=4), np.expand_dims(y_array, axis=4)

def load_neg_data(data_files, mask_files, num_cases, num_samples_per_case, 
	size=(32,32,32)):
	start_time = datetime.datetime.now() 
	print "Loading {} segments from {} negative cases ...".format(num_samples_per_case*num_cases, num_cases)
	X_case_list = [] 
	X2_case_list = [] 
	y_case_list = [] 
	indices = np.random.choice(range(len(data_files)), num_cases)
	pad_list = [(size[0],size[0]),(size[1],size[1]),(size[2],size[2])]
	for index in indices:
		data = nib.load(data_files[index]).get_data()
		mask = nib.load(mask_files[index]).get_data() 
		mask[mask > 0] = 1. 
		data = np.pad(data, pad_list, "constant",
			constant_values=np.min(data))
		gt = np.zeros_like(data)
		mask = np.pad(mask, pad_list, "constant", 
			constant_values=np.min(mask))
		X,X2,y = get_X_y(data, gt, num_samples_per_case, size, mask)
		X_case_list.append(X) ; X2_case_list.append(X2); y_case_list.append(y) 
	X_array = np.vstack(X_case_list)
	X2_array = np.vstack(X2_case_list)
	y_array = np.vstack(y_case_list) 
	print("DONE in {} !".format(datetime.datetime.now() - start_time))	
	return np.expand_dims(X_array, axis=4), np.expand_dims(X2_array, axis=4), np.expand_dims(y_array, axis=4)

def crop_center(img,cropx,cropy,cropz):
    x,y,z = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    startz = z//2-(cropz//2) 
    return img[startx:startx+cropx,starty:starty+cropy,startz:startz+cropz]

def get_X_y(data, gt, num_samples, size=(32,32,32), mask=None):
	if mask is not None: 
		sample_coords = np.column_stack(np.where(mask))
	else: 
		sample_coords = np.column_stack(np.where(gt))
	my_samples = np.random.choice(range(len(sample_coords)), num_samples)
	X = np.empty((num_samples, size[0],size[1],size[2]))
	X2 = np.empty(X.shape)
	y = np.empty((num_samples, size[0]/2,size[1]/2,size[2]/2))
	downsample = block_reduce(data, block_size=(2,2,2), func=np.max) 
	for e, samp_index in enumerate(my_samples):
		samp = sample_coords[samp_index]
		start_x = samp[0] - size[0] / 2 
		end_x = samp[0] + size[0] / 2 
		start_y = samp[1] - size[1] / 2 
		end_y = samp[1] + size[1] / 2 
		start_z = samp[2] - size[2] / 2
		end_z = samp[2] + size[2] / 2 
		start_x2 = samp[0] // 2 - size[0] / 2 
		end_x2 = samp[0] // 2 + size[0] / 2 
		start_y2 = samp[1] // 2 - size[1] / 2 
		end_y2 = samp[1] // 2 + size[1] / 2 
		start_z2 = samp[2] // 2 - size[2] / 2
		end_z2 = samp[2] // 2 + size[2] / 2 		
		X[e] = data[start_x:end_x,start_y:end_y,start_z:end_z]
		X2[e] = downsample[start_x2:end_x2,start_y2:end_y2,start_z2:end_z2]
		y[e] = crop_center(gt[start_x:end_x,start_y:end_y,start_z:end_z],16,16,16)
	return X, X2, y	

# Data generator is too slow
#
# def data_generator(data_files, gt_files, batch_size=4, size=(64,64,32)): 
# 	if len(data_files) != len(gt_files):
# 		raise Exception("Number of data files is not equal to number of ground truth files.")
# 	while 1:
# 		X = np.empty((batch_size,size[0],size[1],size[2]))
# 		y = np.empty(X.shape) 
# 		for _ in range(batch_size):
# 			index = np.random.choice(range(len(data_files)))
# 			data = nib.load(data_files[index]).get_data()
# 			gt = nib.load(gt_files[index]).get_data()
# 			data = np.pad(data, pad_list, "constant",
# 				constant_values=np.min(data))
# 			gt = np.pad(gt, pad_list, "constant")
# 			X[_], y[_] = get_X_y(data, gt, size) 
# 		yield np.expand_dims(X, axis=4), np.expand_dims(y, axis=4) 

import subprocess

pos_data_dir = "/gpfs/data/dmerck/ipan/deepmedic/data/pos/nii/bet/std/"
neg_data_dir = "/gpfs/data/dmerck/ipan/deepmedic/data/neg/nii/bet/std/"
neg_mask_dir = "/gpfs/data/dmerck/ipan/deepmedic/data/neg/mask/"
gt_dir = "/gpfs/data/dmerck/ipan/deepmedic/data/pos/gt/roi/"

pos_data_files = subprocess.check_output("ls " + pos_data_dir, shell=True).split() 
neg_data_files = subprocess.check_output("ls " + neg_data_dir, shell=True).split() 
neg_mask_files = subprocess.check_output("ls " + neg_mask_dir, shell=True).split() 
gt_files = subprocess.check_output("ls " + gt_dir, shell=True).split() 

pos_data_files = [pos_data_dir + _ for _ in pos_data_files]
neg_data_files = [neg_data_dir + _ for _ in neg_data_files] 
neg_mask_files = [neg_mask_dir + _ for _ in neg_mask_files] 
gt_files = [gt_dir + _ for _ in gt_files]

from keras import regularizers 
model = load_wnet((32,32,32,1), kernel_regularizer=regularizers.l2(5e-4))

subepochs = 1000 
num_cases_per_subepoch = 200 
num_samples_per_case = 20 
batch_size = 20 

for sube in range(subepochs):
	pos_data = load_pos_data(pos_data_files, gt_files, num_cases_per_subepoch / 2, 
		num_samples_per_case)
	neg_data = load_neg_data(neg_data_files, neg_mask_files, num_cases_per_subepoch / 2, 
		num_samples_per_case) 
	X = np.vstack((pos_data[0], neg_data[0]))
	X2 = np.vstack((pos_data[1], neg_data[1]))
	y = np.vstack((pos_data[2], neg_data[2]))
	permute_indices = np.random.permutation(range(X.shape[0]))
	X = X[permute_indices]
	X2 = X2[permute_indices]
	y = y[permute_indices]
	model.fit([X,X2],y, batch_size=batch_size, epochs=1, shuffle=True)
