#preprocess.py
import pickle
import os
import numpy as np
import nibabel as nib

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
    
    mris = []
    for modal in modalities:
        mri = np.array(nib_load(os.path.join(path,'skull_strip', modal + '.nii.gz')), dtype='float32', order='C')
        mris.append(mri)

    all_mris = np.stack(mris, -1)  # [240,240,155]

    mask = all_mris.sum(-1) > 0
    for k in range(4):

        x = all_mris[..., k]  #
        y = x[mask]

        # 0.8885
        x[mask] -= y.mean()
        x[mask] /= y.std()

        all_mris[..., k] = x

    with open(os.path.join(path,'4D_data_f32b0.pkl'), 'wb') as f:
        print(os.path.join(path,'4D_data_f32b0.pkl'))

        if has_label:
            pickle.dump((all_mris, label), f)
        else:
            pickle.dump(all_mris, f)

    for i, modal in enumerate(modalities):
        output = os.path.join(path, f"{modal}_f32b0.pkl")
        with open(output, "wb") as f:
            pickle.dump(all_mris[..., i], f)

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
    # doit(test_set)


