"""
Run `organize_tbi_data.py` first to get the NIfTI files and an
organized data directory.
"""
import sys ; sys.path.insert(0, "/users/ipan/scratch/tbinet/src/")
from preprocessing import * 

import nibabel as nib, numpy as np, os, datetime 
import skimage.morphology as morph 

from subprocess import check_output
from skimage.measure import block_reduce 

# Existing directories
# --------------------
edh_data_dir = "/users/ipan/scratch/tbi_data/slicer/edh/ct/nii/"
sdh_data_dir = "/users/ipan/scratch/tbi_data/slicer/sdh/ct/nii/"
iph_data_dir = "/users/ipan/scratch/tbi_data/slicer/iph/ct/nii/"
cont_data_dir = "/users/ipan/scratch/tbi_data/slicer/cont/ct/nii/"

edh_seg_dir = "/users/ipan/scratch/tbi_data/slicer/edh/gt/nii/"
sdh_seg_dir = "/users/ipan/scratch/tbi_data/slicer/sdh/gt/nii/"
iph_seg_dir = "/users/ipan/scratch/tbi_data/slicer/iph/gt/nii/"
cont_seg_dir = "/users/ipan/scratch/tbi_data/slicer/cont/gt/nii/"

# Step 1: New directories
# -----------------------
resampled_edh_data = "/users/ipan/scratch/tbi_data/resampled/edh/ct/"
resampled_sdh_data = "/users/ipan/scratch/tbi_data/resampled/sdh/ct/"
resampled_iph_data = "/users/ipan/scratch/tbi_data/resampled/iph/ct/"
resampled_cont_data = "/users/ipan/scratch/tbi_data/resampled/cont/ct/"

resampled_edh_seg = "/users/ipan/scratch/tbi_data/resampled/edh/gt/"
resampled_sdh_seg = "/users/ipan/scratch/tbi_data/resampled/sdh/gt/"
resampled_iph_seg = "/users/ipan/scratch/tbi_data/resampled/iph/gt/"
resampled_cont_seg = "/users/ipan/scratch/tbi_data/resampled/cont/gt/"

sampling_masks_edh = "/users/ipan/scratch/tbi_data/resampled/edh/masks/"
sampling_masks_sdh = "/users/ipan/scratch/tbi_data/resampled/sdh/masks/"
sampling_masks_iph = "/users/ipan/scratch/tbi_data/resampled/iph/masks/"
sampling_masks_cont = "/users/ipan/scratch/tbi_data/resampled/cont/masks/"

# Step 2: New directories 
# -----------------------
brain_edh_data = "/users/ipan/scratch/tbi_data/final3/bet/edh/"
brain_sdh_data = "/users/ipan/scratch/tbi_data/final3/bet/sdh/"
brain_iph_data = "/users/ipan/scratch/tbi_data/final3/bet/iph/"
brain_cont_data = "/users/ipan/scratch/tbi_data/final3/bet/cont/"

brain_edh_std_data = "/users/ipan/scratch/tbi_data/final3/bet/std/edh/"
brain_sdh_std_data = "/users/ipan/scratch/tbi_data/final3/bet/std/sdh/"
brain_iph_std_data = "/users/ipan/scratch/tbi_data/final3/bet/std/iph/"
brain_cont_std_data = "/users/ipan/scratch/tbi_data/final3/bet/std/cont/"

sampling_mask_edh = "/users/ipan/scratch/tbi_data/final3/masks/edh/"
sampling_mask_sdh = "/users/ipan/scratch/tbi_data/final3/masks/sdh/"
sampling_mask_iph = "/users/ipan/scratch/tbi_data/final3/masks/iph/"
sampling_mask_cont = "/users/ipan/scratch/tbi_data/final3/masks/cont/"

# Step 3: New directories
# -----------------------
edh_new_seg_dir = "/users/ipan/scratch/tbi_data/final3/gt/roi/edh/"
sdh_new_seg_dir = "/users/ipan/scratch/tbi_data/final3/gt/roi/sdh/"
iph_new_seg_dir = "/users/ipan/scratch/tbi_data/final3/gt/roi/iph/"
cont_new_seg_dir = "/users/ipan/scratch/tbi_data/final3/gt/roi/cont/"

# ===== STEP 1 ===== 
# Resample to 0.5 mm x 0.5 mm x 1.0 mm 
def resample_everything(data_dir, seg_dir, new_data_out, new_seg_out):
	if not os.path.exists(new_data_out): os.makedirs(new_data_out)
	if not os.path.exists(new_seg_out): os.makedirs(new_seg_out)
	ct_files = check_output("ls " + data_dir, shell=True).split() 
	gt_files = check_output("ls " + seg_dir, shell=True).split()
	skip = ("161","493","667","722","766","840","1055","1165","1254") 
	for ct in ct_files:
		if ct.split(".")[0] in skip: continue
		start_time = datetime.datetime.now() 
		print "Resampling {} CT data ...".format(ct) 
		ct_nifti = nib.load(os.path.join(data_dir, ct))
		resampled_ct = resample_NIfTI(ct_nifti, [0.5,0.5,1.])
		resampled_ct = rotate_180_degrees(resampled_ct) 
		nib.save(resampled_ct, os.path.join(new_data_out, ct))
		print "DONE in {} !".format(datetime.datetime.now() - start_time) 
	for gt in gt_files: 
		if gt.split(".")[0] in skip: continue
		start_time = datetime.datetime.now() 
		print "Resampling {} segmentation data ...".format(gt)
		gt_nifti = nib.load(os.path.join(seg_dir, gt))
		resampled_gt = resample_NIfTI(gt_nifti, [0.5,0.5,1.], binary=True) 
		resampled_gt = rotate_180_degrees(resampled_gt) 
		nib.save(resampled_gt, os.path.join(new_seg_out, gt))
		print "DONE in {} !".format(datetime.datetime.now() - start_time) 

resample_everything(edh_data_dir, edh_seg_dir, 
	resampled_edh_data, resampled_edh_seg)

resample_everything(sdh_data_dir, sdh_seg_dir, 
	resampled_sdh_data, resampled_sdh_seg)

resample_everything(iph_data_dir, iph_seg_dir, 
	resampled_iph_data, resampled_iph_seg)

resample_everything(cont_data_dir, cont_seg_dir, 
	resampled_cont_data, resampled_cont_seg)

# ===== STEP 2 ===== 
# a. Extract brain using FSL BET 
# b. Max-pooling to 1 mm^3
# c. Thresholding to [0, 100] HU 
# d. Standardizing to zero mean, unary variance
def finish_preprocessing(data_dir, new_data_out, new_mask_out, std_data_out): 
	if not os.path.exists(new_data_out): os.makedirs(new_data_out)
	if not os.path.exists(new_mask_out): os.makedirs(new_mask_out)
	if not os.path.exists(std_data_out): os.makedirs(std_data_out)
	ct_files = check_output("ls " + data_dir, shell=True).split() 
	for ct in ct_files:
		if ct in ("493"): continue 
		start_time = datetime.datetime.now() 
		print "Extracting brain from CT {} ...".format(ct) 
		ct_nifti = nib.load(os.path.join(data_dir, ct))
		brain_ct = extract_brain(ct_nifti) 
		hu_min, hu_max = 0, 100
		print "Thresholding HU values to range [{}, {}] ...".format(hu_min, hu_max)
		brain_ct = threshold_ct(brain_ct, hu_min, hu_max) 
		# Original dims should be 0.5 mm x 0.5 mm x 1.0 mm 
		print "Max-pooling ..."
		brain_ct = max_pool_ct(brain_ct, (2,2,1))
		print "Dilating ..." 
		brain_ct = erode_ct(brain_ct)
		brain_ct = smooth_ct(brain_ct) 
		nib.save(brain_ct, os.path.join(new_data_out, ct))
		# Plan is to use bleed segmentation as sampling mask for positives
		# But make this one anyways just in case you want to use it later
		hu_threshold = 35 
		print "Creating sampling mask using HU threshold {} ...".format(hu_threshold) 
		sampling_mask = np.zeros_like(brain_ct.get_data())
		sampling_mask[brain_ct.get_data() >= hu_threshold] = 1 
		sampling_mask = nib.Nifti1Image(sampling_mask.astype("int16"), 
			np.diag((1,1,1,1)))
		nib.save(sampling_mask, os.path.join(new_mask_out, ct)) 
		print "Standardizing ..."
		brain_mask = brain_ct.get_data().copy() 
		brain_mask[brain_mask > np.min(brain_mask)] = 1
		brain_ct = standardize_ct(brain_ct, brain_mask)
		nib.save(brain_ct, os.path.join(std_data_out, ct))
		print "DONE in {} !".format(datetime.datetime.now() - start_time)

finish_preprocessing(resampled_edh_data, brain_edh_data, 
	sampling_mask_edh, brain_edh_std_data)

finish_preprocessing(resampled_sdh_data, brain_sdh_data, 
	sampling_mask_sdh, brain_sdh_std_data)

finish_preprocessing(resampled_iph_data, brain_iph_data, 
	sampling_mask_iph, brain_iph_std_data)

finish_preprocessing(resampled_cont_data, brain_cont_data, 
	sampling_mask_cont, brain_cont_std_data)

# ==== STEP 3 ==== 
# Edit the ground truth so that none of it lies outside of the extracted brain
# This is to help the network learn better 
def edit_ground_truth(data_dir, seg_dir, out_dir): 
	if not os.path.exists(out_dir): os.makedirs(out_dir)
	gt_files = check_output("ls " + seg_dir, shell=True).split() 
	for gt in gt_files: 
		print "Editing ground truth for {} ...".format(gt) 
		gt_nifti = nib.load(os.path.join(seg_dir, gt)) 
		gt_nifti = max_pool_ct(gt_nifti, (2,2,1)) 
		ct_nifti = nib.load(os.path.join(data_dir, gt))
		gt_array = gt_nifti.get_data()
		orig_sum = np.sum(gt_array) 
		ct_array = ct_nifti.get_data() 
		gt_array[ct_array == 0] = 0 
		pct_reduction = 1 - float(np.sum(gt_array)) / np.sum(orig_sum) + 1e-6
		pct_reduction *= 100 ; pct_reduction = round(pct_reduction, 2) 
		abs_reduction = np.sum(orig_sum) - np.sum(gt_array)
		# This may be exaggerated since the segmentation was resampled 
		print "Segmentation volume reduced by {}% ({} px)".format(pct_reduction, 
			abs_reduction) 
		new_gt_nifti = nib.Nifti1Image(gt_array.astype("int16"), 
			np.diag((1,1,1,1)))
		nib.save(new_gt_nifti, os.path.join(out_dir, gt)) 

edit_ground_truth(brain_edh_data, resampled_edh_seg, 
	edh_new_seg_dir)

edit_ground_truth(brain_sdh_data, resampled_sdh_seg, 
	sdh_new_seg_dir)

edit_ground_truth(brain_iph_data, resampled_iph_seg, 
	iph_new_seg_dir)

edit_ground_truth(brain_cont_data, resampled_cont_seg, 
	cont_new_seg_dir)

	




import nibabel as nib, numpy as np 
from skimage.measure import label 
from scipy.ndimage.filters import median_filter 

x = nib.load("662.nii.gz")
x = x.get_data() 
x[x < 0] = 0 ; x[x > 100] = 0 
x = median_filter(x, (4,4,2)) 
bw = x > 20 
y = label(bw, connectivity=2) 
labels, counts = np.unique(y, return_counts=True) 
labels, counts = labels[1:], counts[1:] 
brain_label = labels[list(counts).index(np.max(counts))]
z = x.copy() 
z[y != brain_label] = np.min(z) 
nib.save(nib.Nifti1Image(z, np.diag((1,1,1,1))), "test.nii.gz")