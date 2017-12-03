"""
This script will convert the CT scans from DICOM files to NIfTI files. 
It also preprocesses all the data. 
"""

import re, os, nrrd, dicom, subprocess
import skimage.morphology as morph 
import nibabel as nib 
import numpy as np 

from scipy import ndimage

from skimage import measure  


def load_DICOM(ddir): 
	"""
	Each CT scan should have its own directory with all of the DICOM files 
	representing slices in that scan. The file names should also be such 
	that they are in order by name. (i.e., slice 1 is <PREFIX>_001, slice 2
	is <PREFIX>_002, or something to that effect)

	Make sure that the DICOM files have file extension .dcm

	This will also return the voxel size as a tuple (x,y,z). 
	"""
	dicom_files = subprocess.check_output("ls " + ddir, shell=True).split() 
	dicom_files = [_ for _ in dicom_files if bool(re.search(r".dcm$", _))]
	# DICOM files should have arrays of size 512 x 512 
	dicom_array = np.empty((512, 512, len(dicom_files)))
	for z in range(len(dicom_files)):
		path_to_dcm = os.path.join(ddir, dicom_files[z])
		tmp_dicom = dicom.read_file(path_to_dcm) 
		dicom_array[:,:,z] = tmp_dicom.pixel_array 
	# Ensure correct typing 
	dicom_array = dicom_array.astype("float32") 
	# Convert to Hounsfield units 
	dicom_array *= int(tmp_dicom.RescaleSlope)
	dicom_array += int(tmp_dicom.RescaleIntercept) 
	return dicom_array

def load_NRRD(ndir, ct=False): 
	"""
	Load NRRD files from Slicer 
	"""
	grep_nrrd_file = "ls {} | grep nrrd".format(ndir) 
	nrrd_files = subprocess.check_output(grep_nrrd_file, shell=True)
	nrrd_files = nrrd_files.split("\n")
	data_exists = False 
	# Find the segmentation file first
	for n in nrrd_files: 
		check = bool(re.search("label.nrrd", n))
		if check: 
			data = n 
			data_exists = True
			continue 
	# If you want the CT data, just grab the other NRRD file 
	if ct and data_exists: 
		nrrd_files.remove(data) 
		data = nrrd_files[0]
	if not data_exists:
		print("Cannot find file !")  
		return None 
	try: 
		nrrd_array, options = nrrd.read(os.path.join(ndir, data))
	except IOError: 
		print "No NRRD CT data file exists !"
		return None 
	return nrrd_array

def convert_DICOM_to_NIfTI(ddir, outdir, filename=None):
	"""
	Uses dcm2niix command line tool to convert DICOM series into NIfTI file. 
	Make sure dcm2niix is in your $PATH.

	Options are pre-specified, might add ability to customize options later on. 
	"""
	if filename == None: filename = os.path.basename(ddir)
	if not os.path.exists(outdir): os.makedirs(outdir) 
	dcm2niix = "dcm2niix -m y -z y -f {} -o {} {}".format(filename, outdir, ddir)
	os.system(dcm2niix) 

def convert_segmentations_to_DICOM(ddir, segmentation, outdir): 
	"""
	Given DICOM directory and segmentation for that specific CT scan,
	turn the segmentation into DICOM files.

	Segmentation should be output from load_segmentations func.
	"""
	if not os.path.exists(outdir): os.makedirs(outdir)
	dicom_files = subprocess.check_output("ls " + ddir, shell=True).split() 
	dicom_files = [_ for _ in dicom_files if bool(re.search(r".dcm$", _))]
	if len(dicom_files) != segmentation.shape[2]:
		print "Number of DICOM files does not match number of slices in segmentation"
		print "DICOM : {} // Segmentation : {}".format(len(dicom_files), segmentation.shape[2])
		return None 
	# Make sure it's binary 
	segmentation[segmentation > 0] = 1 
	for z in range(len(dicom_files)):
		path_to_dcm = os.path.join(ddir, dicom_files[z])
		tmp_dicom = dicom.read_file(path_to_dcm) 
		seg_dicom = tmp_dicom 
		seg_dicom.PixelData = segmentation[:,:,z].tostring()
		seg_dicom.RescaleIntercept = 0 
		seg_dicom.RescaleSlope = 1
		seg_dicom.save_as(os.path.join(outdir, dicom_files[z]))

def resample_NIfTI(nii, newdim, binary=False): 
	"""
	Takes in a NIfTI file from nibabel and resamples it to the new dimension
	"""
	pixdim = nii._header["pixdim"]
	old_x = pixdim[1] ; old_y = pixdim[2] ; old_z = pixdim[3]
    new_x = newdim[0] ; new_y = newdim[1] ; new_z = newdim[2]
    resize_x = old_x / new_x 
    resize_y = old_y / new_y
    resize_z = old_z / new_z 
    array = nii.get_data()
    if binary: assert len(np.unique(array)) == 2
    original_min = np.min(array) ; original_max = np.max(array)
    rs_array = ndimage.interpolation.zoom(array, [resize_x,resize_y,resize_z], 
    	order=1, mode="nearest", prefilter=False)
    rs_array[rs_array < original_min] = original_min 
    rs_array[rs_array > original_max] = original_max 
    if binary:
    	rs_array[rs_array >= 0.5] = 1 
    	rs_array[rs_array < 0.5] = 0
    	assert len(np.unique(rs_array)) == 2 
    else:
    	rs_array = np.round(rs_array)
    # Turn it back into a nibabel object
    new_nii = nib.Nifti1Image(rs_array.astype("int16"), np.diag((1,1,1,1)))
    new_nii._header["pixdim"][1:4] = list(newdim) 
    return new_nii 

def pad_volume(nii, newdim): 
	"""
	Pads a volume with zeros to the desired dimension. 
	Takes as input nibabel object. 
	"""
	array = nii.get_data() 
	# Calculate size differences in each plane
	x_diff = newdim[0] - array.shape[0]
	y_diff = newdim[1] - array.shape[1] 
	z_diff = newdim[2] - array.shape[2] 
	# If for some reason you put in a larger/same-sized volume
	# and the difference calculated is negative, make it 0.
	# It will return the same size as the input. 
	if x_diff < 0: x_diff = 0
	if y_diff < 0: y_diff = 0 
	if z_diff < 0: z_diff = 0
	# Determine padding 
	x_pad = (x_diff // 2, x_diff - x_diff // 2)
	y_pad = (y_diff // 2, y_diff - y_diff // 2)
	z_pad = (z_diff // 2, z_diff - z_diff // 2)
	pad_list = [x_pad, y_pad, z_pad] 
	# Make the new array 
	new_array = np.pad(array, pad_list, "constant", 
		constant_values=np.min(array))
	new_nii = nib.Nifti1Image(new_array, np.diag((1,1,1,1))) 
	return new_nii 

def extract_brain(nii): 
	"""
	Use FSL BET to extract brain. 
	Takes as input nibabel object. Returns new nibabel object. 
	"""
	array = nii.get_data()
	down_array = measure.block_reduce(array, (2,2,1), np.max)
	smoothed_array = ndimage.filters.gaussian_filter(down_array, sigma=1)
	# Threshold out skull 
	# Everything outside [-50, 100] is set to 50 HU 
	smoothed_array[smoothed_array < -50] = -50
	smoothed_array[smoothed_array > 100] = -50 
	original_size = array.shape 
	# Max-pooling to 1 mm^3 
	# Smoothing to improve quality of brain extraction
	randint = np.random.randint(1e6)
	tmp_filename = ".tmp-{}.nii.gz".format(randint)
	tmp_nii = nib.Nifti1Image(smoothed_array, np.diag((1,1,1,1)))
	nib.save(tmp_nii, tmp_filename) 
	out_prefix = ".bet-{}".format(randint)
	tmp_maskname = ".bet-{}_mask.nii.gz".format(randint)
	bet_command = "bet2 {} {} -f 0.01 -m".format(tmp_filename, out_prefix)
	os.system(bet_command) 
	bet_mask = nib.load(tmp_maskname).get_data()
	# Upsample bet_mask=
	up_x = float(original_size[0]) / bet_mask.shape[0]
	up_y = float(original_size[1]) / bet_mask.shape[1]
	up_z = float(original_size[2]) / bet_mask.shape[2] 
	bet_mask = ndimage.interpolation.zoom(bet_mask, [up_x,up_y,up_z], order=1,
		mode="nearest", prefilter=False)
	bet_mask = np.round(bet_mask).astype("uint8")
	bet_mask = ndimage.binary_fill_holes(bet_mask)
	# Probably not necessary to use remove_small_objects
	bet_mask = measure.label(bet_mask, connectivity=2)
	if np.max(bet_mask) > 1: 
		bet_mask = morph.remove_small_objects(bet_mask, min_size=100, connectivity=2)
	array[bet_mask == 0] = np.min(array)
	os.system("rm {} {}.nii.gz {}".format(tmp_filename, out_prefix, tmp_maskname))
	return nib.Nifti1Image(array, np.diag((1,1,1,1)))

def threshold_ct(nii, hu_min, hu_max):
	"""
	All voxels outside of [hu_min, hu_max] are set to hu_min
	"""
	array = nii.get_data() 
	array[array < hu_min] = hu_min 
	array[array > hu_max] = hu_min 
	return nib.Nifti1Image(array, np.diag((1,1,1,1)))

def standardize_ct(nii, mask):
	"""
	Standardizes Hounsfield units in CT scan to have within-image
	0 mean, 1 standard deviation
	"""
	array = nii.get_data().astype("float32")
	array_mean = np.mean(array[mask == 1]) 
	array_sd = np.std(array[mask == 1]) 
	array -= array_mean 
	array /= array_sd 
	return nib.Nifti1Image(array.astype("float32"), np.diag((1,1,1,1)))

def max_pool_ct(nii, size):
	array = nii.get_data() 
	array = measure.block_reduce(array, size, np.max) 
	return nib.Nifti1Image(array, np.diag((1,1,1,1)))

def erode_ct(nii, mask=None): 
	array = nii.get_data() 
	if mask == None: 
		mask = np.zeros_like(array) 
		mask[array > np.min(array)] = 1 
	mask = morph.binary_erosion(mask, morph.ball(1))
	array[mask == 0] = np.min(array) 
	return nib.Nifti1Image(array, np.diag((1,1,1,1)))

def dilate_ct(nii, mask=None): 
	array = nii.get_data() 
	if mask == None: 
		mask = np.zeros_like(array) 
		mask[array > np.min(array)] = 1 
	mask = morph.binary_dilation(mask, morph.ball(3))
	array[mask == 0] = np.min(array) 
	return nib.Nifti1Image(array, np.diag((1,1,1,1)))

def smooth_ct(nii, filter_size=(3,3,3)): 
	array = nii.get_data() 
	array = ndimage.filters.median_filter(array, filter_size)
	return nib.Nifti1Image(array, np.diag((1,1,1,1))) 

def rotate_180_degrees(nii): 
	array = nii.get_data() 
	for _ in range(array.shape[2]): 
		array[:,:,_] = np.rot90(array[:,:,_], k=2) 
	new_nii = nib.Nifti1Image(array, np.diag((1,1,1,1)))
	for _ in new_nii._header: 
		new_nii._header[_] = nii._header[_]
	return new_nii 

