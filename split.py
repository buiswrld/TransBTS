#split.py

# make the python lst lst (with all the paths of the DeepBraTumIA MRI scans and segmentation masks)
import os
from random import shuffle
from math import floor

root = "/lambda/nfs/KAMS/Imaging"

# list will the full path to each set of MRI set's atlas folder
subject_paths = []

# Walk through directory tree
for dirpath, dirnames, filenames in os.walk(root):

  # if the folder contains MRI scans or segmentation masks
  if any(f.endswith("_skull_strip.nii.gz") for f in filenames) or "seg_mask.nii.gz" in filenames:
    atlas_folder = os.path.dirname(dirpath)
    path_without_root = os.path.relpath(atlas_folder, root)

    if path_without_root not in subject_paths:
      subject_paths.append(path_without_root)

# take in an input called lst which is just a python list, each element being a string of a filename (or example, element 0 could be "file0.nii")
# take in an input called proportion which will dictate the size of the test & validation set

def txt_split(lst: list, proportion: float):
    # --- basic input checks ---
  if not isinstance(lst, list):
      print('Invalid list!')

    # --- make a copy so we don't change the original list ---
  data = lst.copy()

    # --- shuffle the list so the split is random ---
  shuffle(data)

    # --- figure out how many items go to each split ---
  N = len(data)
  val_n  = floor(proportion * N)

    # --- slice the shuffled list into three parts (no overlap) ---
  val_set  = data[0: val_n]
    # do same/similar for train set
  train_set = data[val_n : ]

    # --- write each split to its own .txt file (one path per line) ---
  with open('train.txt', 'w') as f:
    f.writelines(item + '\n' for item in train_set)
  with open('valid.txt', 'w') as f:
    f.writelines(item + '\n' for item in val_set)

    # do same for other sets

    # --- optionally: print some stats at the end so we can read it and make sure it happened correctly ---
  with open('train.txt', 'r') as f:
      train_lines = f.readlines()  # read all lines into a list
  print(f"First 5 lines of train.txt: {train_lines[:5]}; Total lines: {len(train_lines)}\n")

  with open('valid.txt', 'r') as f:
      val_lines = f.readlines()
  print(f"First 5 lines of valid.txt: {val_lines[:5]}; Total lines: {len(val_lines)}\n")

txt_split(subject_paths, 0.3)