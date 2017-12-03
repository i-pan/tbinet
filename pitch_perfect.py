#######################
# CREATING PREDICTORS #
####################### 

# Assume img is a 3D numpy array 

# Intensity 
intensity = (img >= 40) & (img <= 80) 

# Set up the convolve filter for easy averaging 
cube3 = np.full((3,3,3), 1./27)

# Mean-filtered image 
mean_filt = scipy.ndimage.filters.convolve(img, cub3)
mean_filt[mask == 0] = 0 

# Local standard deviation 
local_sd = scipy.ndimage.filters.convolve((img-mean_filt) ** 2, cube3)
local_sd = np.sqrt(local_sd) 
local_sd[mask == 0] = 0 

# Skew 
skew = scipy.ndimage.filters.convolve((img-mean_filt) ** 3, cube3)
skew = skew / (local_sd + 1e-7) ** 3
skew[mask == 0] = 0

# Kurtosis
kurt = scipy.ndimage.filters.convolve((img-mean_filt) ** 4, cube3) 
kurt = kurt / (local_sd + 1e-7) ** 4
kurt[mask == 0] = 0 

# Percentage of neighbors with intensities in certain range
percent = scipy.ndimage.filters.convolve(intensity.astype("float32"), cube3) 
percent[mask == 0] = 0 

# For ICH, authors used percentage of neighborhood voxels equal to 0 
# We don't use this because epidural/subdurals are along the brain surface
# This means one edge will have many neighbors = 0 whereas the other edge 
# will not

def within_plane_std_score(img, mask, plane, windsorized=False): 
	mean_list = [] ; std_list = [] 
	if plane == "x": 
		dim = img.shape[0] 
	elif plane == "y": 
		dim = img.shape[1] 
	elif plane == "z": 
		dim = img.shape[2]
	else:
		raise Exception("Plane <{}> is not valid".format(plane))
	for each_slice in range(dim):
		if plane == "x": 
			mask_slice = mask[each_slice,:,:]
			img_slice = img[each_slice,:,:]
		elif plane == "y": 
			mask_slice = mask[:,each_slice,:]
			img_slice = img[:,each_slice,:]
		elif plane == "z": 
			mask_slice = mask[:,:,each_slice]
			img_slice = img[:,:,each_slice]
		if windsorized: 
			perc20 = np.percentile(img_slice[mask_slice != 0], 20)
			perc80 = np.percentile(img_slice[mask_slice != 0], 80)
			img_slice[img_slice < perc20] = perc20 
			img_slice[img_slice > perc80] = perc80
		if np.sum(mask_slice) == 0: 
			mean_list.append(0) 
			std_list.append(1e-7)
		else: 
			mean_list.append(np.mean(img_slice[mask_slice != 0]))
			std_list.append(np.std(img_slice[mask_slice != 0]))
	z_score = img.copy() 
	for each_slice in range(dim): 
		if plane == "x": 
			z_score[each_slice,:,:] -= mean_list[each_slice]
			z_score[each_slice,:,:] /= std_list[each_slice]
		elif plane == "y": 
			z_score[:,each_slice,:] -= mean_list[each_slice]
			z_score[:,each_slice,:] /= std_list[each_slice]
		elif plane == "z": 
			z_score[:,:,each_slice] -= mean_list[each_slice]
			z_score[:,:,each_slice] /= std_list[each_slice]
	z_score[mask == 0] = np.min(z_score) 
	return z_score

# Calculate distance from center of brain 
centroid = np.asarray(scipy.ndimage.measurements.center_of_mass(mask))
coords = np.column_stack(np.where(mask != 0))
distance = img.copy() 
for coo in coords:
	distance[coo[0],coo[1],coo[2]] = np.linalg.norm(coo-centroid)

# Gaussian smoothing 


# Within-x-plane standardized score 
x_mean = [] ; x_std = [] 
for each_x_slice in range(img.shape[0]):
	mask_slice = mask[each_x_slice,:,:]
	if np.sum(mask_slice) == 0: 
		x_mean.append(0) 
		x_std.append(1e-7)
	else: 
		x_mean.append(np.mean(img[each_x_slice,:,:][mask_slice != 0]))
		x_std.append(np.std(img[each_x_slice,:,:][mask_slice != 0]))

x_score = img.copy() 
for each_x_slice in range(img.shape[0]):
	x_score[each_x_slice,:,:] = (x_score[each_x_slice,:,:] - x_mean[each_x_slice]) / x_std[each_x_slice]


x_score[mask == 0] = np.min(x_score)

# Within-y-plane standardized score 
y_mean = [] ; y_std = [] 
for each_y_slice in range(img.shape[1]):
	mask_slice = mask[:,each_y_slice,:]
	if np.sum(mask_slice) == 0: 
		y_mean.append(0) 
		y_std.append(1e-7)
	else: 
		y_mean.append(np.mean(img[:,each_y_slice,:][mask_slice != 0]))
		y_std.append(np.std(img[:,each_y_slice,:][mask_slice != 0]))

y_score = img.copy() 
for each_y_slice in range(img.shape[1]):
	y_score[:,each_y_slice,:] = (y_score[:,each_y_slice,:] - y_mean[each_y_slice]) / y_std[each_y_slice]


y_score[mask == 0] = np.min(y_score)

# Within-z-plane standardized score 
z_mean = [] ; z_std = [] 
for each_z_slice in range(img.shape[2]):
	mask_slice = mask[:,:,each_z_slice]
	if np.sum(mask_slice) == 0: 
		z_mean.append(0) 
		z_std.append(1e-7)
	else: 
		z_mean.append(np.mean(img[:,:,each_z_slice][mask_slice != 0]))
		z_std.append(np.std(img[:,:,each_z_slice][mask_slice != 0]))

z_score = img.copy() 
for each_z_slice in range(img.shape[2]):
	z_score[:,:,each_z_slice] = (z_score[:,:,each_z_slice] - z_mean[each_z_slice]) / z_std[each_z_slice]


z_score[mask == 0] = np.min(z_score)

# Within-x-plane standardized score - WINDSORIZED
x_mean = [] ; x_std = [] 
img_w = img.copy() 
for each_x_slice in range(img.shape[0]):
	mask_slice = mask[each_x_slice,:,:]
	if np.sum(mask_slice) == 0: 
		x_mean.append(0) 
		x_std.append(1e-7)
	else: 
		img_w_slice = img_w[each_x_slice,:,:]
		perc20 = np.percentile(img_w_slice[mask_slice != 0], 20)
		perc80 = np.percentile(img_w_slice[mask_slice != 0], 80)
		img_w_slice[img_w_slice < perc20] = perc20 
		img_w_slice[img_w_slice > perc80] = perc80
		x_mean.append(np.mean(img_w_slice[mask_slice != 0]))
		x_std.append(np.std(img_w_slice[mask_slice != 0]))

x_w_score = img.copy() 
for each_x_slice in range(img.shape[0]):
	x_w_score[each_x_slice,:,:] = (x_w_score[each_x_slice,:,:] - x_mean[each_x_slice]) / x_std[each_x_slice]