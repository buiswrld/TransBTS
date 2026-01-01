#preprocess.py
import pickle
import os
import numpy as np
import nibabel as nib

from BraTS import enforce_shape

modalities = ('flair_skull_strip', 'ct1_skull_strip', 't1_skull_strip', 't2_skull_strip')

# train
train_set = {
        'root': '/lambda/nfs/KAMS/TransBTS',
        'flist': 'train.txt',
        'has_label': True
        }

# test/validation data
valid_set = {
        'root': '/lambda/nfs/KAMS/TransBTS',
        'flist': 'valid.txt',
        'has_label': False
        }

test_set = {
        'root': '/lambda/nfs/KAMS/TransBTS',
        'flist': 'test.txt',
        'has_label': False
        }


def nib_load(file_name):
    if not os.path.exists(file_name):
        print('Invalid file name, can not find the file!')

    proxy = nib.load(file_name)
    data = proxy.get_fdata()
    proxy.uncache()
    return data


def process_i16(path, has_label=True):
    """ Save the original 3D MRI all_mris with dtype=int16.
        Noted that no normalization is used! """


    label = np.array(nib_load(os.path.join(path, 'segmentation', 'seg_mask.nii.gz')), dtype='uint8', order='C')

    all_mris = np.stack([
        np.array(nib_load(os.path.join(path,'skull_strip', modal + '.nii.gz')), dtype='int16', order='C')
        for modal in modalities], -1)# [240,240,155]

    output = os.path.join(path,'data_i16.pkl')

    with open(output, 'wb') as f:
        print(output)
        print(all_mris.shape, type(all_mris), label.shape, type(label))  # (240,240,155,4) , (240,240,155)
        pickle.dump((all_mris, label), f)

    if not has_label:
        return


def process_f32b0(path, has_label=True):
    """ Save the data with dtype=float32.
        z-score is used but keep the background with zero! """
    if has_label:
        label = np.array(nib_load(os.path.join(path, 'segmentation', 'seg_mask.nii.gz')), dtype='uint8', order='C')
        label = enforce_shape(label)

    images = []
    for modal in modalities:
        img = np.array(nib_load(os.path.join(path,'skull_strip', modal + '.nii.gz')), dtype='float32', order='C')
        img = enforce_shape(img)
        images.append(img)

    images = np.stack(images, -1)  # (240,240,155,4)


    output = os.path.join(path, 'data_f32b0.pkl')
    mask = images.sum(-1) > 0
    for k in range(4):

        x = images[..., k]  #
        y = x[mask]

        # 0.8885
        x[mask] -= y.mean()
        x[mask] /= y.std()

        images[..., k] = x

    assert images.shape[:3] == (240, 240, 155)

    with open(output, 'wb') as f:
        print(output)
        if has_label:
            pickle.dump((images, label), f)
        else:
            pickle.dump(images, f)

    if not has_label:
        return


def doit(dset):
    root, has_label = dset['root'], dset['has_label']
    file_list = os.path.join(root, dset['flist'])
    subjects = open(file_list).read().splitlines()
    paths = [os.path.join('/lambda/nfs/KAMS/Imaging', sub) for sub in subjects]

    for path in paths:

        process_f32b0(path, has_label)



if __name__ == '__main__':
    doit(train_set)
    doit(valid_set)
    doit(test_set)




