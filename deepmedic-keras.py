import keras 

from keras import backend as K, optimizers
from keras.engine import Input, Model 
from keras.layers import Conv3D, Conv3DTranspose, BatchNormalization
from keras.layers import Add, Concatenate, Cropping3D, Activation, Dropout

def check_input_shapes(local_input_shape, global_input_shape,
	downsample_factor):
	local_end_shape = [local_input_shape[0]-16, 
		local_input_shape[1]-16, 
		local_input_shape[2]-16]
	global_end_shape = [global_input_shape[0]-16,
		global_input_shape[1]-16,
		global_input_shape[2]-16]
	global_end_shape = [downsample_factor*_ for _ in global_end_shape]
	if local_end_shape != global_end_shape:
		raise Exception("Local and global input shapes are incompatible")

# No one does this - it's always two convs before adding the residual
#
# def double_conv_residual(num_fms, x, filter_size, 
# 	kernel_initializer, kernel_regularizer, dropout): 
# 	# Two convolutional layers
# 	x = Conv3D(num_fms, filter_size, activation="relu", padding="valid", 
# 		kernel_initializer=kernel_initializer,
# 		kernel_regularizer=kernel_regularizer)(x) 
# 	if dropout is not None: 
# 		x = Dropout(dropout)(x) 
# 	x = BatchNormalization()(x) 
# 	# No ReLU for the 2nd 
# 	y = Conv3D(num_fms, filter_size, padding="valid", 
# 		kernel_initializer=kernel_initializer,
# 		kernel_regularizer=kernel_regularizer)(x) 
# 	if dropout is not None: 
# 		x = Dropout(dropout)(x) 
# 	y = BatchNormalization()(y) 
# 	# Crop the FMs so that the dims are equal 
# 	r = Cropping3D(cropping=((1,1),(1,1),(1,1)))(x) 
# 	# Residual
# 	y = Add()([y, r])
# 	# Now apply ReLU 
# 	y = Activation("relu")(y) 
# 	return y 

def residual_block(num_fms, x, filter_size, 
	kernel_initializer, kernel_regularizer, dropout): 
	x = Conv3D(num_fms, filter_size, activation="relu", padding="valid",
		kernel_initializer=kernel_initializer,
		kernel_regularizer=kernel_regularizer)(x) 
	x = BatchNormalization()(x) 
	if dropout is not None: 
		x = Dropout(dropout)(x) 
	y = Conv3D(num_fms, filter_size, activation="relu", padding="valid",
		kernel_initializer=kernel_initializer,
		kernel_regularizer=kernel_regularizer)(x) 
	if dropout is not None: 
		x = Dropout(dropout)(x) 
	y = BatchNormalization()(y) 
	y = Conv3D(num_fms, filter_size, padding="valid",
		kernel_initializer=kernel_initializer,
		kernel_regularizer=kernel_regularizer)(y) 
	if dropout is not None: 
		x = Dropout(dropout)(x) 
	y = BatchNormalization()(y) 
	r = Cropping3D(cropping=((2,2),(2,2),(2,2)))(x) 
	y = Add()([y, r]) 
	y = Activation("relu")(y) 
	return y 

def load_deepmedic(local_input_shape=(25,25,25,1), 
	global_input_shape=(19,19,19,1), downsample_factor=3, 
	filter_size=(3,3,3), n_classes=1,
	kernel_initializer="he_normal", kernel_regularizer=None,
	dropout_conv=None, dropout_fc=None):
	check_input_shapes(local_input_shape, global_input_shape, downsample_factor)
	local_input = Input(local_input_shape) 
	global_input = Input(global_input_shape)
	# Initial double conv, no dropout 
	l_conv_init = Conv3D(30, filter_size, activation="relu", padding="valid",
		kernel_initializer=kernel_initializer,
		kernel_regularizer=kernel_regularizer)(local_input) 
	l_conv_init = BatchNormalization()(l_conv_init) 
	l_conv_init = Conv3D(30, filter_size, activation="relu", padding="valid",
		kernel_initializer=kernel_initializer,
		kernel_regularizer=kernel_regularizer)(l_conv_init) 
	l_conv_init = BatchNormalization()(l_conv_init) 
	l_residual1 = residual_block(40, l_conv_init, filter_size, 
		kernel_initializer=kernel_initializer,
		kernel_regularizer=kernel_regularizer, dropout=dropout_conv)
	l_residual2 = residual_block(50, l_residual1, filter_size,
		kernel_initializer=kernel_initializer,
		kernel_regularizer=kernel_regularizer, dropout=dropout_conv)
	g_conv_init = Conv3D(30, filter_size, activation="relu", padding="valid",
		kernel_initializer=kernel_initializer,
		kernel_regularizer=kernel_regularizer)(global_input) 
	g_conv_init = BatchNormalization()(g_conv_init) 
	g_conv_init = Conv3D(30, filter_size, activation="relu", padding="valid",
		kernel_initializer=kernel_initializer,
		kernel_regularizer=kernel_regularizer)(g_conv_init) 
	g_conv_init = BatchNormalization()(g_conv_init) 
	g_residual1 = residual_block(40, g_conv_init, filter_size, 
		kernel_initializer=kernel_initializer,
		kernel_regularizer=kernel_regularizer, dropout=dropout_conv)
	g_residual2 = residual_block(50, g_residual1, filter_size,
		kernel_initializer=kernel_initializer,
		kernel_regularizer=kernel_regularizer, dropout=dropout_conv)
	# Upsample global pathway output to match local pathway output
	g_deconvolu = Conv3DTranspose(50, filter_size, strides=(3,3,3),
		activation="relu", padding="valid", 
		kernel_initializer=kernel_initializer,
		kernel_regularizer=kernel_regularizer)(g_residual2) 
	g_deconvolu = BatchNormalization()(g_deconvolu)
	# Concatenate
	concat = Concatenate(axis=4)([l_residual2, g_deconvolu]) 
	fc09 = Conv3D(150, (1,1,1), activation="relu", padding="valid")(concat) 
	fc09 = BatchNormalization()(fc09)
	fc10 = Conv3D(150, (1,1,1), padding="valid")(fc09) 
	if dropout_fc is not None: 
		fc10 = Dropout(dropout_fc)(fc10) 
	fc10 = BatchNormalization()(fc10)
	fc10 = Add()([fc09, fc10])
	fc10 = Activation("relu")(fc10)
	if n_classes == 1:
		out_activation = "sigmoid"
	elif n_classes > 1: 
		out_activation = "softmax"
	predictions = Conv3D(n_classes, (1,1,1), activation=out_activation, 
		padding="valid")(fc10) 
	if dropout_fc is not None: 
		predictions = Dropout(dropout_fc)(predictions) 
	model = Model(inputs=[local_input, global_input], outputs=predictions)
	return model

def dense_block(num_fms, x, filter_size, 
	kernel_initializer, kernel_regularizer, dropout): 
	"""
	4-layer dense block 
	Each layer is connected to the layer before it 
	In the DenseNet paper, they specify the following order:
		BN --> ReLU --> Conv on the concatenation
	"""
	# Layer 1
	d1 = BatchNormalization()(x) 
	d1 = Activation("relu")(d1) 
	d1 = Conv3D(num_fms, filter_size, padding="valid",
		kernel_initializer=kernel_initializer,
		kernel_regularizer=kernel_regularizer)(d1) 
	if dropout is not None: 
		d1 = Dropout(dropout)(d1) 
	# Layer 2 
	d2 = BatchNormalization()(d1) 
	d2 = Activation("relu")(d2) 
	d2 = Conv3D(num_fms, filter_size, padding="valid",
		kernel_initializer=kernel_initializer,
		kernel_regularizer=kernel_regularizer)(d2) 
	if dropout is not None: 
		d2 = Dropout(dropout)(d2) 
	# Layer 3
	# Concatenate layers 1, 2, output
	r1 = Cropping3D(cropping=((1,1),(1,1),(1,1)))(d1) 
	d3 = Concatenate()([r1, d2]) 
	d3 = BatchNormalization()(d3) 
	d3 = Activation("relu")(d3) 
	d3 = Conv3D(num_fms, filter_size, padding="valid",
		kernel_initializer=kernel_initializer,
		kernel_regularizer=kernel_regularizer)(d3) 
	if dropout is not None: 
		d3 = Dropout(dropout)(d3)
	# Layer 4
	# Concatenate layers 1, 2, 3 output
	r1 = Cropping3D(cropping=((2,2),(2,2),(2,2)))(d1) 
	r2 = Cropping3D(cropping=((1,1),(1,1),(1,1)))(d2) 
	d4 = Concatenate(axis=4)([r1, r2, d3])
	d4 = BatchNormalization()(d4)  
	d4 = Activation("relu")(d4) 
	d4 = Conv3D(num_fms, filter_size, padding="valid",
		kernel_initializer=kernel_initializer,
		kernel_regularizer=kernel_regularizer)(d4) 
	if dropout is not None: 
		d4 = Dropout(dropout)(d4)
	return d4 

def load_densemedic(local_input_shape=(25,25,25,1), 
	global_input_shape=(19,19,19,1), downsample_factor=3, 
	filter_size=(3,3,3), n_classes=1,
	kernel_initializer="he_normal", kernel_regularizer=None,
	dropout_conv=None, dropout_fc=None):
	local_input = Input(local_input_shape) 
	global_input = Input(global_input_shape)
	# Initial conv, no dropout 
	l_conv_init = Conv3D(30, filter_size, padding="same",
		kernel_initializer=kernel_initializer,
		kernel_regularizer=kernel_regularizer)(local_input) 
	# First 4-layer dense block 
	l_dense1 = dense_block(40, l_conv_init, filter_size,
		kernel_initializer=kernel_initializer,
		kernel_regularizer=kernel_regularizer, dropout=dropout_conv) 
	l_conv2 = Conv3D(50, filter_size, padding="same",
		kernel_initializer=kernel_initializer,
		kernel_regularizer=kernel_regularizer)(l_dense1) 
	l_dense2 = dense_block(50, l_conv2, filter_size,
		kernel_initializer=kernel_initializer,
		kernel_regularizer=kernel_regularizer, dropout=dropout_conv)
	# Initial conv, no dropout 
	g_conv_init = Conv3D(30, filter_size, padding="same",
		kernel_initializer=kernel_initializer,
		kernel_regularizer=kernel_regularizer)(global_input) 
	# First 4-layer dense block 
	g_dense1 = dense_block(40, g_conv_init, filter_size,
		kernel_initializer=kernel_initializer,
		kernel_regularizer=kernel_regularizer, dropout=dropout_conv) 
	g_conv2 = Conv3D(50, filter_size, padding="same",
		kernel_initializer=kernel_initializer,
		kernel_regularizer=kernel_regularizer)(g_dense1) 
	g_dense2 = dense_block(50, g_conv2, filter_size,
		kernel_initializer=kernel_initializer,
		kernel_regularizer=kernel_regularizer, dropout=dropout_conv)
	g_upsamp = Conv3DTranspose(50, filter_size, strides=(3,3,3),
		activation="relu", padding="valid", 
		kernel_initializer=kernel_initializer,
		kernel_regularizer=kernel_regularizer)(g_dense2) 
	g_upsamp = BatchNormalization()(g_upsamp)
	concat = Concatenate(axis=4)([l_dense2, g_upsamp]) 
	fc09 = Conv3D(150, (1,1,1), activation="relu", padding="valid")(concat) 
	fc09 = BatchNormalization()(fc09)
	fc10 = Conv3D(150, (1,1,1), padding="valid")(fc09) 
	if dropout_fc is not None: 
		fc10 = Dropout(dropout_fc)(fc10) 
	fc10 = Add()([fc09, fc10])
	fc10 = BatchNormalization()(fc10)
	fc10 = Activation("relu")(fc10)
	if n_classes == 1:
		out_activation = "sigmoid"
	elif n_classes > 1: 
		out_activation = "softmax"
	predictions = Conv3D(n_classes, (1,1,1), activation=out_activation, 
		padding="valid")(fc10) 
	if dropout_fc is not None: 
		predictions = Dropout(dropout_fc)(predictions) 
	model = Model(inputs=[local_input, global_input], outputs=predictions)
	return model












