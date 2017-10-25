from keras import layers 
from keras import optimizers
from keras import regularizers
from keras import backend as K 
from keras.models import load_model
from keras.engine import Input, Model 
from keras.layers import Conv3D, Conv3DTranspose, BatchNormalization, Dropout

import datetime 
import subprocess
import numpy as np 
import nibabel as nib 
import multiprocessing as mp 

from skimage.measure import block_reduce 


################################
# === SETTING UP THE MODEL === #
################################

def double_conv_bn(x, num_fm, filter_size, activation="relu", padding="same", 
    kernel_initializer="he_normal", kernel_regularizer=None):
    """
    Shortcut for 2 batch-normalized Conv3D layers that will be used often 
    """
    conv = Conv3D(num_fm, filter_size, activation=activation, padding=padding,
        kernel_initializer=kernel_initializer, 
        kernel_regularizer=kernel_regularizer)(x) 
    conv = BatchNormalization()(conv) 
    conv = Conv3D(num_fm, filter_size, activation=activation, padding=padding,
        kernel_initializer=kernel_initializer, 
        kernel_regularizer=kernel_regularizer)(conv) 
    conv = BatchNormalization()(conv)
    return conv 

def local_wnet(inputs, filter_size, activation="relu", padding="same", 
    kernel_initializer="he_normal", kernel_regularizer=None):
    """
    W-Net for local input (original segment resolution) 
    """
    conv1 = double_conv_bn(inputs, 32, filter_size, 
        activation=activation,
        padding=padding, 
        kernel_initializer=kernel_initializer, 
        kernel_regularizer=kernel_regularizer)
    # Double # of feature maps when downsampling with conv layer 
    down1 = Conv3D(64, (2,2,2), activation=activation, padding=padding, 
        strides=(2,2,2), 
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer)(conv1) 
    down1 = BatchNormalization()(down1) 
    #
    conv2 = double_conv_bn(down1, 64, filter_size, 
        activation=activation,
        padding=padding, 
        kernel_initializer=kernel_initializer, 
        kernel_regularizer=kernel_regularizer)
    # Residual connections from input to above double conv layer 
    conv2 = layers.Add()([down1, conv2])
    down2 = Conv3D(128, (2,2,2), activation=activation, padding=padding, 
        strides=(2,2,2), 
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer)(conv2) 
    down2 = BatchNormalization()(down2) 
    # Assuming 32^3 input, should now be 8^3 so start deconvolutions 
    # after final double conv layers 
    conv3 = double_conv_bn(down2, 128, filter_size, 
        activation=activation,
        padding=padding, 
        kernel_initializer=kernel_initializer, 
        kernel_regularizer=kernel_regularizer)
    conv3 = layers.Add()([down2, conv3])
    up3 = Conv3DTranspose(128, (2,2,2), activation=activation, padding=padding,
        strides=(2,2,2),
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer)(conv3)
    up3 = BatchNormalization()(up3) 
    # Concatenate output after conv layers from previous layer with the same
    # output size 
    # Assuming 32^3 input, this would be the layer with size 16^3
    local_output = layers.Concatenate(axis=4)([conv2, up3])
    return local_output 

def global_wnet(inputs, filter_size, activation="relu", padding="same", 
    kernel_initializer="he_normal", kernel_regularizer=None):
    """
    W-Net for global input (1/2 original segment resolution) 
    Differences between local and global pathways:
    - (1) Start with 24 feature maps 
    - (2) One extra downsampling step 
    """
    conv1 = double_conv_bn(inputs, 24, filter_size, 
        activation=activation,
        padding=padding, 
        kernel_initializer=kernel_initializer, 
        kernel_regularizer=kernel_regularizer)
    # Double # of feature maps when downsampling with conv layer 
    down1 = Conv3D(48, (2,2,2), activation=activation, padding=padding, 
        strides=(2,2,2), 
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer)(conv1) 
    down1 = BatchNormalization()(down1) 
    #
    conv2 = double_conv_bn(down1, 48, filter_size, 
        activation=activation,
        padding=padding, 
        kernel_initializer=kernel_initializer, 
        kernel_regularizer=kernel_regularizer)
    # Residual connections from input to above double conv layer 
    conv2 = layers.Add()([down1, conv2])
    down2 = Conv3D(96, (2,2,2), activation=activation, padding=padding, 
        strides=(2,2,2), 
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer)(conv2) 
    down2 = BatchNormalization()(down2)  
    conv3 = double_conv_bn(down2, 96, filter_size, 
        activation=activation,
        padding=padding, 
        kernel_initializer=kernel_initializer, 
        kernel_regularizer=kernel_regularizer)
    conv3 = layers.Add()([down2, conv3])
    down3 = Conv3D(192, (2,2,2), activation=activation, padding=padding, 
        strides=(2,2,2), 
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer)(conv3) 
    down3 = BatchNormalization()(down3) 
    # With 32^3 input, input is now 4^3 so start deconvolution after
    conv4 = double_conv_bn(down3, 192, filter_size, 
        activation=activation,
        padding=padding, 
        kernel_initializer=kernel_initializer, 
        kernel_regularizer=kernel_regularizer)
    conv4 = layers.Add()([down3, conv4])
    up4 = Conv3DTranspose(192, (2,2,2), activation=activation, padding=padding,
        strides=(2,2,2),
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer)(conv4)
    up4 = BatchNormalization()(up4) 
    conv5 = double_conv_bn(up4, 192, filter_size, 
        activation=activation,
        padding=padding, 
        kernel_initializer=kernel_initializer, 
        kernel_regularizer=kernel_regularizer)
    conv5 = layers.Add()([up4, conv5])
    # Halve # of feature maps at the next deconvolution
    up5 = Conv3DTranspose(96, (2,2,2), activation=activation, padding=padding,
        strides=(2,2,2),
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer)(conv5) 
    up5 = BatchNormalization()(up5) 
    global_output = layers.Concatenate(axis=4)([conv2, up5])
    return global_output 

def load_wnet(optimizer, input_shape=(32,32,32,1), filter_size=(3,3,3), 
    n_classes=1, 
    loss="binary_crossentropy", metrics=["accuracy"], 
    activation="relu", padding="same", 
    kernel_initializer="he_normal", kernel_regularizer=None): 
    """
    Full W-Net combines local and global pathways
    For binary classes, n_classes can be 1 or 2 
    **Note: All kernels have the same regularizer 
    """
    local_input = Input(input_shape) ; global_input = Input(input_shape) 
    local_output = local_wnet(local_input, filter_size, activation, padding,
        kernel_initializer, kernel_regularizer)
    global_output = global_wnet(global_input, filter_size, activation, padding,
        kernel_initializer, kernel_regularizer)
    combined_output = layers.Concatenate(axis=4)([local_output, global_output])
    #fc1 = Conv3D(256, (1,1,1), activation=activation, padding=padding,
    #    kernel_initializer=kernel_initializer, 
    #    kernel_regularizer=kernel_regularizer)(combined_output)
    #fc1 = BatchNormalization()(fc1) 
    #fc2 = Conv3D(256, (1,1,1), activation=activation, padding=padding,
    #    kernel_initializer=kernel_initializer,
    #    kernel_regularizer=kernel_regularizer)(fc1) 
    #fc2 = BatchNormalization()(fc2) 
    # (1,1,1) filter with sigmoid/softmax to convert to segmentation 
    if n_classes > 1: 
        last_activation = "softmax"
    else: 
        last_activation = "sigmoid"
    predictions = Conv3D(n_classes, (1,1,1), activation=last_activation, 
        padding="same", 
        kernel_initializer=kernel_initializer, 
        kernel_regularizer=kernel_regularizer)(combined_output)
    model = Model(inputs=[local_input, global_input], 
        outputs=predictions)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model 

########################
# === LOADING DATA === #
########################

def crop_center(img,cropx,cropy,cropz):
    """
    Crops out the center of a 3D volume
    """
    x,y,z = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    startz = z//2-(cropz//2) 
    return img[startx:startx+cropx,starty:starty+cropy,startz:startz+cropz]

def get_X_y(data, gt, num_samples, size=(32,32,32), mask=None):
    """
    Samples from a given case
    For positives, it will use the ground truth mask to sample positive segments
    For negatives, it will use the sampling mask (specify mask for negatives)
    """
    if mask is not None: 
        sample_coords = np.column_stack(np.where(mask))
    else: 
        sample_coords = np.column_stack(np.where(gt))
    my_samples = np.random.choice(range(len(sample_coords)), num_samples)
    Xl = np.empty((num_samples, size[0],size[1],size[2]))
    Xg = np.empty(Xl.shape)
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
        Xl[e] = data[start_x:end_x,start_y:end_y,start_z:end_z]
        Xg[e] = downsample[start_x2:end_x2,start_y2:end_y2,start_z2:end_z2]
        y[e] = crop_center(gt[start_x:end_x,start_y:end_y,start_z:end_z],
                    16,16,16)
    return Xl, Xg, y 

def load_pos_data(queue, data_files, gt_files, 
    num_cases, num_samples_per_case, 
    size=(32,32,32)):
    """
    Given a set of cases, it will load arrays of sample segments. 
    """
    start_time = datetime.datetime.now() 
    total_samples = num_samples_per_case*num_cases
    print "Loading {} segments from {} positive cases ...".format(total_samples,
        num_cases)
    Xl_case_list = [] 
    Xg_case_list = [] 
    y_case_list = [] 
    indices = np.random.choice(range(len(data_files)), num_cases)
    pad_list = [(size[0],size[0]),(size[1],size[1]),(size[2],size[2])]
    for index in indices:
        data = nib.load(data_files[index]).get_data()
        gt = nib.load(gt_files[index]).get_data()
        data = np.pad(data, pad_list, "constant",
            constant_values=np.min(data))
        gt = np.pad(gt, pad_list, "constant")
        Xl,Xg,y = get_X_y(data, gt, num_samples_per_case, size)
        Xl_case_list.append(Xl) ; Xg_case_list.append(Xg); y_case_list.append(y) 
    Xl_array = np.vstack(Xl_case_list)
    Xg_array = np.vstack(Xg_case_list)
    y_array = np.vstack(y_case_list) 
    Xl_array = np.expand_dims(Xl_array, axis=4) 
    Xg_array = np.expand_dims(Xg_array, axis=4)
    y_array = np.expand_dims(y_array, axis=4)
    print("\nDONE loading positives in {} !\n".format(datetime.datetime.now() - start_time))
    queue.put((Xl_array, Xg_array, y_array))

def load_neg_data(queue, data_files, mask_files, 
    num_cases, num_samples_per_case, 
    size=(32,32,32)):
    """
    Given a set of cases, it will load arrays of sample segments. 
    """
    start_time = datetime.datetime.now() 
    total_samples = num_samples_per_case*num_cases
    print "Loading {} segments from {} negative cases ...".format(total_samples,
        num_cases)
    Xl_case_list = [] 
    Xg_case_list = [] 
    y_case_list = [] 
    indices = np.random.choice(range(len(data_files)), num_cases)
    pad_list = [(size[0],size[0]),(size[1],size[1]),(size[2],size[2])]
    for index in indices:
        data = nib.load(data_files[index]).get_data()
        mask = nib.load(mask_files[index]).get_data()
        mask[mask > 0] = 1. 
        data = np.pad(data, pad_list, "constant",
            constant_values=np.min(data))
        mask = np.pad(mask, pad_list, "constant",
            constant_values=np.min(mask))
        gt = np.zeros_like(data)
        # Remember to specify mask!
        Xl,Xg,y = get_X_y(data, gt, num_samples_per_case, size, mask)
        Xl_case_list.append(Xl) ; Xg_case_list.append(Xg); y_case_list.append(y) 
    Xl_array = np.vstack(Xl_case_list)
    Xg_array = np.vstack(Xg_case_list)
    y_array = np.vstack(y_case_list) 
    Xl_array = np.expand_dims(Xl_array, axis=4) 
    Xg_array = np.expand_dims(Xg_array, axis=4)
    y_array = np.expand_dims(y_array, axis=4)
    print("\nDONE loading negatives in {} !\n".format(datetime.datetime.now() - start_time))
    queue.put((Xl_array, Xg_array, y_array))

def load_pos_data_parallel(index): 
    """
    Requires to be defined in the environment: 
        pos_data_files 
        gt_files 
        num_samples_per_case
        size 
    """
    pad_list = [(size[0],size[0]),(size[1],size[1]),(size[2],size[2])]
    data = nib.load(pos_data_files[index]).get_data() 
    gt = nib.load(gt_files[index]).get_data() 
    data = np.pad(data, pad_list, "constant",
                constant_values=np.min(data))
    gt = np.pad(gt, pad_list, "constant")
    Xl,Xg,y = get_X_y(data, gt, num_samples_per_case, size)
    Xl = np.expand_dims(Xl, axis=4)
    Xg = np.expand_dims(Xg, axis=4)
    y = np.expand_dims(y, axis=4)
    return Xl, Xg, y

def load_neg_data_parallel(index): 
    """
    Requires to be defined in the environment: 
        neg_data_files 
        mask_files 
        num_samples_per_case
        size 
    """
    pad_list = [(size[0],size[0]),(size[1],size[1]),(size[2],size[2])]
    data = nib.load(neg_data_files[index]).get_data()
    mask = nib.load(neg_mask_files[index]).get_data() 
    mask[mask > 0] = 1. 
    data = np.pad(data, pad_list, "constant",
        constant_values=np.min(data))
    gt = np.zeros_like(data)
    mask = np.pad(mask, pad_list, "constant", 
        constant_values=np.min(mask))
    Xl,Xg,y = get_X_y(data, gt, num_samples_per_case, size, mask)
    Xl = np.expand_dims(Xl, axis=4)
    Xg = np.expand_dims(Xg, axis=4)
    y = np.expand_dims(y, axis=4)
    return Xl, Xg, y

def make_array_from_list(data_list, size=(32,32,32)):
    num_samples = data_list[0][0].shape[0]
    Xl = np.empty((len(data_list)*num_samples, size[0],size[1],size[2], 1))
    Xg = np.empty(Xl.shape)
    y = np.empty((Xl.shape[0],size[0]/2,size[1]/2,size[2]/2, 1))
    j = 0 
    for i, _ in enumerate(data_list):
        Xl[j:j+num_samples] = data_list[i][0]
        Xg[j:j+num_samples] = data_list[i][1]
        y[j:j+num_samples] = data_list[i][2]
        j += num_samples
    return Xl, Xg, y

###################
# === EXECUTE === #
###################

# **NOTE: Verify whether loading in parallel or loading on single thread 
#         during training is faster ... 

# --- DEFINE ALL THESE VARIABLES AS IS --- 
# Or else parallel loading scripts won't work ... 
size = (32,32,32) 
num_samples_per_case = 20

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
# --- END VARIABLES --- 

num_cases_per_subepoch = 50 
batch_size = 20 

# Load the data for the initial subepoch 
pos_indices = np.random.choice(range(len(pos_data_files)), 
    num_cases_per_subepoch)
neg_indices = np.random.choice(range(len(neg_data_files)), 
    num_cases_per_subepoch)

if __name__ == "__main__":
    num_processes = 4 
    p = mp.Pool(processes=num_processes)
    start_time = datetime.datetime.now() 
    total_samples = num_samples_per_case*num_cases_per_subepoch
    print "Loading {} segments from {} positive cases ...".format(total_samples,
        num_cases_per_subepoch)    
    pos_data_list = p.map(load_pos_data_parallel, pos_indices) 
    print "DONE in {} !".format(datetime.datetime.now() - start_time)
    start_time = datetime.datetime.now() 
    print "Loading {} segments from {} negative cases ...".format(total_samples,
        num_cases_per_subepoch)    
    neg_data_list = p.map(load_neg_data_parallel, neg_indices)
    print "DONE in {} !".format(datetime.datetime.now() - start_time)
    p.close()

# Need to turn the lists from Pool into arrays 
Xl_pos, Xg_pos, y_pos = make_array_from_list(pos_data_list, size)
Xl_neg, Xg_neg, y_neg = make_array_from_list(neg_data_list, size)
Xl = np.vstack((Xl_pos, Xl_neg)) ; del Xl_pos, Xl_neg
Xg = np.vstack((Xg_pos, Xg_neg)) ; del Xg_pos, Xg_neg 
y = np.vstack((y_pos, y_neg)) ; del y_pos, y_neg 

permute_indices = np.random.permutation(range(Xl.shape[0]))
Xl = Xl[permute_indices]
Xg = Xg[permute_indices]
y = y[permute_indices]

# Define a function for Keras model fitting to pass to process 
def fit_model(queue, model_name, Xl, Xg, y):
    """
    Saves the Keras model to a temp file which you need to load later 
    to resume training. This is because Queue cannot properly pickle Keras 
    object. 
    """
    model.fit([Xl,Xg],y, batch_size=batch_size, epochs=1, shuffle=True)
    filepath = ".keras_{}.hdf5".format(model_name) 
    model.save(filepath)
    queue.put(filepath) 

# Load your model
model = load_wnet(optimizers.Adam())

# Spawn your processes 
subepochs = 100 
for sube in range(1, subepochs): 
    if __name__ == "__main__":
        manager = mp.Manager() 
        q = manager.Queue() 
        pos_data = mp.Process(target=load_pos_data, 
            args=(q,pos_data_files,gt_files,num_cases_per_subepoch,
                    num_samples_per_case))
        neg_data = mp.Process(target=load_neg_data, 
            args=(q,neg_data_files,neg_mask_files,num_cases_per_subepoch,
                    num_samples_per_case))
	processes = [pos_data, neg_data]
        for p in processes: p.start()
        model.fit([Xl, Xg], y, batch_size=batch_size, epochs=1, shuffle=True)
        for p in processes: p.join() 
        results = [q.get() for p in processes] 
    # Now I have a trained model, loaded data for next subepoch
    # So now:
    # (1) Load the model 
    # (2) Save the model weights
    # (3) Set up the data again 
    # Remember to save weights here 
    del Xl, Xg, y 
    Xl = np.vstack((results[0][0], results[1][0]))
    Xg = np.vstack((results[0][1], results[1][1]))
    y = np.vstack((results[0][2], results[1][2]))
    permute_indices = np.random.permutation(range(Xl.shape[0]))
    Xl = Xl[permute_indices]
    Xg = Xg[permute_indices]
    y = y[permute_indices]
