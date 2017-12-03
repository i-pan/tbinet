"""
This script restructures and preprocesses the segmentation/DICOM data 
that Owen has prepared.

Modifications made to Owen's directory before processing through this script:
    1. Get rid of spaces in directory names 
    2. In subdurals: "mv 250b-Segmented 1250b-Segmented"
    3. In subdurals: "mv 453b-Segmented 543b-Segmented"
    4. In subdurals: "mv 948\ -\ baseline 948-baseline"
    5. In IPHs - 169: "mv 09038441/* . ; rmdir 09038441/"
    6. In contusions: "mv 1163b-Rethinned/1163b-Segmented ."
    7. In contusions: "mv 1257b-\ Segmented 1257b-Segmented"
    8. In contusions: "mv 745b-Segmented\ / 745b-Segmented/"
"""
import sys ; sys.path.insert(0, "/users/ipan/scratch/tbinet/src/")
from preprocessing import * 

from subprocess import check_output

#############
# POSITIVES #
#############

# ===== STEP 1 ===== # 
# Convert CT DICOM into NIfTI 
# This will provide the voxel dimensions when we use the NRRD file from Slicer
# for the CT data

def get_path_to_CT(dicom_path, _):
    path_to_ct = os.path.join(dicom_path, _, _+"-baseline", _+"b-Rethinned")
    if not os.path.exists(path_to_ct): 
        path_to_ct = os.path.join(dicom_path, _, _+"-Baseline", _+"b-Rethinned")
    if not os.path.exists(path_to_ct): 
        path_to_ct = os.path.join(dicom_path, _, _+"-baseline", _+"b-rethinned")
    if not os.path.exists(path_to_ct): 
        path_to_ct = os.path.join(dicom_path, _, _+"-baseline", _+"b-rethinned")
    if not os.path.exists(path_to_ct): 
        path_to_ct = os.path.join(dicom_path, _, _+"b-Rethinned")
    if not os.path.exists(path_to_ct):
        path_to_ct = os.path.join(dicom_path, _, _+"-baseline", _+"-Rethinned")
    if not os.path.exists(path_to_ct):
        path_to_ct = os.path.join(dicom_path, _, _+"b-baseline", _+"b-Rethinned")
    return path_to_ct 

def get_NIfTI_files_from_DICOM(dicom_path, nii_path, tilt=False): 
    dicom_cases = check_output("ls " + dicom_path, shell=True).split() 
    for _ in dicom_cases:
        path_to_ct = get_path_to_CT(dicom_path, _)
        new_nii_dir = os.path.join(nii_path, _)
        if not os.path.exists(new_nii_dir): os.makedirs(new_nii_dir)
        convert_DICOM_to_NIfTI(path_to_ct, new_nii_dir)
        # Remove one of the files depending on `tilt` argument
        # Rename the file 
        nii_files = check_output("ls " + new_nii_dir, shell=True).split() 
        if len(nii_files) == 1: 
            os.system("mv {}/{} {}/{}".format(new_nii_dir, nii_files[0], 
                new_nii_dir, _+".nii.gz"))
        elif len(nii_files) > 1: 
            length_of_file_names = [len(ni) for ni in nii_files] 
            if tilt:
                remove_this_one = length_of_file_names.index(np.min(length_of_file_names)) 
                keep_this_one = length_of_file_names.index(np.max(length_of_file_names))
            else:
                remove_this_one = length_of_file_names.index(np.max(length_of_file_names)) 
                keep_this_one = length_of_file_names.index(np.min(length_of_file_names))
            os.system("rm {}/{}".format(new_nii_dir, nii_files[remove_this_one]))
            os.system("mv {}/{} {}/{}".format(new_nii_dir, nii_files[keep_this_one], 
                new_nii_dir, _+".nii.gz"))

edh_dicom_path = "/users/ipan/scratch/ProTECT_3D/edh/baseline/"
edh_nii_path = "/users/ipan/scratch/tbi_data/orig/edh/ct/nii/"
sdh_dicom_path = "/users/ipan/scratch/ProTECT_3D/sdh/baseline/"
sdh_nii_path = "/users/ipan/scratch/tbi_data/orig/sdh/ct/nii/"
iph_dicom_path = "/users/ipan/scratch/ProTECT_3D/iph/baseline/"
iph_nii_path = "/users/ipan/scratch/tbi_data/orig/iph/ct/nii/"
cont_dicom_path = "/users/ipan/scratch/ProTECT_3D/cont/baseline/"
cont_nii_path = "/users/ipan/scratch/tbi_data/orig/cont/ct/nii/"

get_NIfTI_files_from_DICOM(edh_dicom_path, edh_nii_path)
get_NIfTI_files_from_DICOM(sdh_dicom_path, sdh_nii_path)
get_NIfTI_files_from_DICOM(iph_dicom_path, iph_nii_path)
get_NIfTI_files_from_DICOM(cont_dicom_path, cont_nii_path)
# Ignore 493 in contusions

# ===== STEP 2 ===== #
# Convert NRRD segmentation + NRRD CT data to NIfTI 

def get_path_to_NRRD_files(nrrd_path, _):
    path_to_nrrd = os.path.join(nrrd_path, _, _+"-baseline", _+"b-Segmented")
    if not os.path.exists(path_to_nrrd): 
        path_to_nrrd = os.path.join(nrrd_path, _, _+"-Baseline", _+"b-Segmented")
    if not os.path.exists(path_to_nrrd): 
        path_to_nrrd = os.path.join(nrrd_path, _, _+"-Baseline", _+"b-segmented")
    if not os.path.exists(path_to_nrrd): 
        path_to_nrrd = os.path.join(nrrd_path, _, _+"-baseline", _+"b-segmented")
    if not os.path.exists(path_to_nrrd): 
        path_to_nrrd = os.path.join(nrrd_path, _, _+"-baseline", _+"b-Segmentation")
    if not os.path.exists(path_to_nrrd): 
        path_to_nrrd = os.path.join(nrrd_path, _, _+"-baseline", _+"-Segmentation")
    if not os.path.exists(path_to_nrrd): 
        path_to_nrrd = os.path.join(nrrd_path, _, _+"-baseline", _+"b-segmented")
    if not os.path.exists(path_to_nrrd): 
        path_to_nrrd = os.path.join(nrrd_path, _, _+"b-Segmented")
    if not os.path.exists(path_to_nrrd): 
        path_to_nrrd = os.path.join(nrrd_path, _, _+"b-segmented")
    if not os.path.exists(path_to_nrrd):
        path_to_nrrd = os.path.join(nrrd_path, _, _+"-baseline", _+"-Segmented")
    if not os.path.exists(path_to_nrrd):
        path_to_nrrd = os.path.join(nrrd_path, _, _+"b-baseline", _+"b-Segmented")
    if not os.path.exists(path_to_nrrd):
        # Segmentation files were just in the directory 
        path_to_nrrd = os.path.join(nrrd_path, _)
    return path_to_nrrd

def get_NIfTI_from_NRRD(nii_path, nrrd_path, ct_out_dir, gt_out_dir,
    cont=False):
    nifti_files = check_output("ls " + nii_path, shell=True).split()
    if not os.path.exists(ct_out_dir): os.makedirs(ct_out_dir)
    if not os.path.exists(gt_out_dir): os.makedirs(gt_out_dir) 
    for _ in nifti_files:
        # 493 has weird top/bottom split
        # 1165, 1254, 766, 840 doesn't have segmentation folder 
        if _ in ("493", "1165", "1254", "766", "840", "948", "1100"): continue
        print "Getting NIfTI CT data // segmentation for {} ...".format(_) 
        path_to_original_nifti = os.path.join(nii_path, _, _+".nii.gz") 
        path_to_nrrd_files = get_path_to_NRRD_files(nrrd_path, _) 
        original_nifti = nib.load(path_to_original_nifti)
        ct_data = load_NRRD(path_to_nrrd_files, ct=True) 
        segmentation = load_NRRD(path_to_nrrd_files, ct=False) 
        if segmentation is None: 
            print "Segmentation not found !"
            continue 
        if ct_data is None: 
            print "CT data not found !"
            gt_nifti = nib.Nifti1Image(segmentation, np.diag((1,1,1,1)))
            gt_nifti._header["pixdim"] = original_nifti._header["pixdim"] 
            nib.save(gt_nifti, os.path.join(gt_out_dir, _+".nii.gz"))        
        # Make sure the dimensions match up 
        if original_nifti.shape[2] != ct_data.shape[2]: 
            print "Original NIfTI scan does not match dimensions of NRRD CT data"
            print "{} // {}".format(original_nifti.shape, ct_data.shape) 
            continue
        if ct_data.shape[2] != segmentation.shape[2]:
            print "Original NIfTI scan does not match dimensions of NRRD CT data"
            print "{} // {}".format(ct_data.shape, segmentation.shape) 
            continue
        if cont: 
            # Get rid of the edema segmentation
            segmentation[ct_data < 35] = 0
        segmentation[segmentation > 0] = 1
        print "Minimum HU : {}".format(np.min(ct_data))
        ct_nifti = nib.Nifti1Image(ct_data, np.diag((1,1,1,1)))
        gt_nifti = nib.Nifti1Image(segmentation, np.diag((1,1,1,1)))
        ct_nifti._header["pixdim"] = original_nifti._header["pixdim"]
        gt_nifti._header["pixdim"] = original_nifti._header["pixdim"] 
        nib.save(ct_nifti, os.path.join(ct_out_dir, _+".nii.gz"))
        nib.save(gt_nifti, os.path.join(gt_out_dir, _+".nii.gz"))

edh_ct_dir = "/users/ipan/scratch/tbi_data/slicer/edh/ct/nii/"
edh_gt_dir = "/users/ipan/scratch/tbi_data/slicer/edh/gt/nii/"
sdh_ct_dir = "/users/ipan/scratch/tbi_data/slicer/sdh/ct/nii/"
sdh_gt_dir = "/users/ipan/scratch/tbi_data/slicer/sdh/gt/nii/"
iph_ct_dir = "/users/ipan/scratch/tbi_data/slicer/iph/ct/nii/"
iph_gt_dir = "/users/ipan/scratch/tbi_data/slicer/iph/gt/nii/"
cont_ct_dir = "/users/ipan/scratch/tbi_data/slicer/cont/ct/nii/"
cont_gt_dir = "/users/ipan/scratch/tbi_data/slicer/cont/gt/nii/"

get_NIfTI_from_NRRD(edh_nii_path, edh_dicom_path, edh_ct_dir, edh_gt_dir)
# 948 in SDH doesn't have the CT data NRRD file
# So I will manually verify that the NIfTI correponds to the ground truth
get_NIfTI_from_NRRD(sdh_nii_path, sdh_dicom_path, sdh_ct_dir, sdh_gt_dir)
get_NIfTI_from_NRRD(iph_nii_path, iph_dicom_path, iph_ct_dir, iph_gt_dir)
# Manually do 1100 in contusions
# Actually the CT data for 1100 is messed up - should probably leave it out  
# Manually do 1234, 1257, 1265, 803, 847, 918
# Check 194 - min HU is 0 
# -- Remove the random label file with a bunch of numbers that has all zeros
get_NIfTI_from_NRRD(cont_nii_path, cont_dicom_path, cont_ct_dir, cont_gt_dir,
    cont=True)

# Code to do it manually
# Make sure you check the orientation first so you can correct it
# accordingly
"""
scp ipan@transfer.ccv.brown.edu:/users/ipan/scratch/tbi_data/orig/cont/gt/nii/918/918.nii.gz 918_seg.nii.gz
scp ipan@transfer.ccv.brown.edu:/users/ipan/scratch/tbi_data/orig/cont/ct/nii/918/918.nii.gz .
"""
import numpy as np, nibabel as nib 
cont = True 
x = nib.load("918.nii.gz")
y = nib.load("918_seg.nii.gz")
xarr = x.get_data() ; yarr = y.get_data() 
if cont: yarr[xarr < 35] = 0 

yarr[yarr > 0] = 1 
for _ in range(xarr.shape[2]): xarr[:,:,_] = np.rot90(xarr[:,:,_], k=2)

for _ in range(yarr.shape[2]): yarr[:,:,_] = np.rot90(yarr[:,:,_], k=2) 

xnib = nib.Nifti1Image(xarr.astype("int16"), np.diag((1,1,1,1)))
ynib = nib.Nifti1Image(yarr, np.diag((1,1,1,1)))
xnib.header["pixdim"] = x.header["pixdim"]
ynib.header["pixdim"] = y.header["pixdim"]
nib.save(xnib, "918.nii.gz")
nib.save(ynib, "918_seg.nii.gz") 
"""
scp 918_seg.nii.gz ipan@transfer.ccv.brown.edu:/users/ipan/scratch/tbi_data/slicer/cont/gt/nii/918.nii.gz
scp 918.nii.gz ipan@transfer.ccv.brown.edu:/users/ipan/scratch/tbi_data/slicer/cont/ct/nii/918.nii.gz
"""

#############
# OLD STUFF #
#############
# Convert your segmentations to DICOM
def get_DICOM_segmentations(dicom_path, out_dir):
    dicom_cases = check_output("ls " + dicom_path, shell=True).split() 
    for _ in dicom_cases:
        # 493 has weird top/bottom split
        # 1165, 1254, 766, 840 doesn't have segmentation folder 
        if _ in ("493", "1165", "1254", "766", "840"): continue
        print "Converting segmentations for {} to DICOM ...".format(_) 
        path_to_ct = get_path_to_CT(dicom_path, _)
        path_to_seg = get_path_to_segmentation(dicom_path, _)
        segmentation = load_segmentations(path_to_seg)
        if segmentation is None: continue 
        new_seg_dir = os.path.join(out_dir, _)
        if not os.path.exists(new_seg_dir): os.makedirs(new_seg_dir)
        convert_segmentations_to_DICOM(path_to_ct, segmentation, new_seg_dir)

edh_dicom_path = "/users/ipan/scratch/ProTECT_3D/edh/baseline/"
edh_seg_path = "/users/ipan/scratch/tbi_data/orig/edh/gt/dicom/"
sdh_dicom_path = "/users/ipan/scratch/ProTECT_3D/sdh/baseline/"
sdh_seg_path = "/users/ipan/scratch/tbi_data/orig/sdh/gt/dicom/"
iph_dicom_path = "/users/ipan/scratch/ProTECT_3D/iph/baseline/"
iph_seg_path = "/users/ipan/scratch/tbi_data/orig/iph/gt/dicom/"
cont_dicom_path = "/users/ipan/scratch/ProTECT_3D/cont/baseline/"
cont_seg_path = "/users/ipan/scratch/tbi_data/orig/cont/gt/dicom/"

get_DICOM_segmentations(edh_dicom_path, edh_seg_path)
get_DICOM_segmentations(sdh_dicom_path, sdh_seg_path)
get_DICOM_segmentations(iph_dicom_path, iph_seg_path)
# 667 had errors 
get_DICOM_segmentations(cont_dicom_path, cont_seg_path)
# 1055, 1165, 1254, 766, 840 missing segmentation
# 161 has errors
# Ignore 493 in contusions

# ===== STEP 3 ===== #
# Convert DICOM segmentations to NIfTI
# 
def get_NIfTI_segmentations(dicom_path, nii_path, tilt=False): 
    dicom_cases = check_output("ls " + dicom_path, shell=True).split() 
    for _ in dicom_cases:
        path_to_ct = os.path.join(dicom_path, _)
        new_nii_dir = os.path.join(nii_path, _)
        if not os.path.exists(new_nii_dir): os.makedirs(new_nii_dir)
        convert_DICOM_to_NIfTI(path_to_ct, new_nii_dir)
        # Remove one of the files depending on `tilt` argument
        # Rename the file 
        nii_files = check_output("ls " + new_nii_dir, shell=True).split() 
        if len(nii_files) > 1: 
            length_of_file_names = [len(ni) for ni in nii_files] 
            if tilt:
                remove_this_one = length_of_file_names.index(np.min(length_of_file_names)) 
                keep_this_one = length_of_file_names.index(np.max(length_of_file_names))
            else:
                remove_this_one = length_of_file_names.index(np.max(length_of_file_names)) 
                keep_this_one = length_of_file_names.index(np.min(length_of_file_names))
            os.system("rm {}/{}".format(new_nii_dir, nii_files[remove_this_one]))

edh_seg_path = "/users/ipan/scratch/tbi_data/orig/edh/gt/dicom/"
edh_nii_path = "/users/ipan/scratch/tbi_data/orig/edh/gt/nii/"
sdh_seg_path = "/users/ipan/scratch/tbi_data/orig/sdh/gt/dicom/"
sdh_nii_path = "/users/ipan/scratch/tbi_data/orig/sdh/gt/nii/"
iph_seg_path = "/users/ipan/scratch/tbi_data/orig/iph/gt/dicom/"
iph_nii_path = "/users/ipan/scratch/tbi_data/orig/iph/gt/nii/"
cont_seg_path = "/users/ipan/scratch/tbi_data/orig/cont/gt/dicom/"
cont_nii_path = "/users/ipan/scratch/tbi_data/orig/cont/gt/nii/"

get_NIfTI_segmentations(edh_seg_path, edh_nii_path)
get_NIfTI_segmentations(sdh_seg_path, sdh_nii_path)
get_NIfTI_segmentations(iph_seg_path, iph_nii_path)
get_NIfTI_segmentations(cont_seg_path, cont_nii_path)

#############
# NEGATIVES #
#############
# Convert DICOM files to NIfTI, preserving directory structure.
#

neg_dicom_path = "/gpfs/data/dmerck/tbi_data/orig/neg/dicom"
neg_nii_path = "/gpfs/data/dmerck/tbi_data/orig/neg/nii"

def batch_DICOM_to_NIfTI(dicom_path, nii_path, tilt=False):
    DICOM_directories = [dicom_path+"/"+directory for directory in os.listdir(dicom_path)]
    for DICOM_directory in DICOM_directories:
        NIfTI_directory = neg_nii_path + "/" + DICOM_directory.split("/")[-1]
        if not os.path.exists(NIfTI_directory):
            os.mkdir(NIfTI_directory)
        convert_DICOM_to_NIfTI(DICOM_directory, NIfTI_directory)

# batch_DICOM_to_NIfTI(neg_dicom_path, neg_nii_path)
