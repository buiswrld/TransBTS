import os
import time
import logging
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import numpy as np
import nibabel as nib
import imageio
import csv

def one_hot(ori, classes):

    batch, h, w, d = ori.size()
    new_gd = torch.zeros((batch, classes, h, w, d), dtype=ori.dtype).cuda()
    for j in range(classes):
        index_list = (ori == j).nonzero()

        for i in range(len(index_list)):
            batch, height, width, depth = index_list[i]
            new_gd[batch, j, height, width, depth] = 1

    return new_gd.float()


def tailor_and_concat(x, model):
    temp = []

    temp.append(x[..., :128, :128, :128])
    temp.append(x[..., :128, 112:240, :128])
    temp.append(x[..., 112:240, :128, :128])
    temp.append(x[..., 112:240, 112:240, :128])
    temp.append(x[..., :128, :128, 27:155])
    temp.append(x[..., :128, 112:240, 27:155])
    temp.append(x[..., 112:240, :128, 27:155])
    temp.append(x[..., 112:240, 112:240, 27:155])

    # run first patch to learn output channel count (num classes)
    out0 = model(temp[0])
    if isinstance(out0, (list, tuple)):  # safety: some models return multiple outputs
        out0 = out0[0]

    # allocate stitched output using model output channels (not input channels)
    y = out0.new_zeros((out0.size(0), out0.size(1), x.size(2), x.size(3), x.size(4)))

    temp[0] = out0
    for i in range(1, len(temp)):
        temp[i] = model(temp[i])
        if isinstance(temp[i], (list, tuple)):
            temp[i] = temp[i][0]

    y[..., :128, :128, :128] = temp[0]
    y[..., :128, 128:240, :128] = temp[1][..., :, 16:128, :]
    y[..., 128:240, :128, :128] = temp[2][..., 16:128, :, :]
    y[..., 128:240, 128:240, :128] = temp[3][..., 16:128, 16:128, :]
    y[..., :128, :128, 128:155] = temp[4][..., 96:123]
    y[..., :128, 128:240, 128:155] = temp[5][..., :, 16:128, 96:123]
    y[..., 128:240, :128, 128:155] = temp[6][..., 16:128, :, 96:123]
    y[..., 128:240, 128:240, 128:155] = temp[7][..., 16:128, 16:128, 96:123]

    return y[..., :155]


def dice_score(o, t, eps=1e-8):
    num = 2*(o*t).sum() + eps
    den = o.sum() + t.sum() + eps
    return num/den


def mIOU(o, t, eps=1e-8):
    num = (o*t).sum() + eps
    # Replaced (o | t)
    den = torch.logical_or(o, t).sum() + eps
    return num/den


def softmax_mIOU_score(output, target):
    mIOU_score = []
    mIOU_score.append(mIOU(o=(output==1),t=(target==1)))
    mIOU_score.append(mIOU(o=(output==2),t=(target==2)))
    mIOU_score.append(mIOU(o=(output==3),t=(target==4)))
    return mIOU_score


def softmax_output_dice(output, target):
    ret = []

    # whole
    o = output > 0; t = target > 0 # ce
    ret += dice_score(o, t),
    # core
    o = (output == 1) | (output == 3)
    t = (target == 1) | (target == 4)
    ret += dice_score(o, t),
    # active
    o = (output == 3);t = (target == 4)
    ret += dice_score(o, t),

    return ret


keys = 'whole', 'core', 'enhancing', 'loss'


def validate_softmax(
        valid_loader,
        model,
        load_file,
        multimodel,
        savepath='',  # when in validation set, you must specify the path to save the 'nii' segmentation results here
        names=None,  # The names of the patients orderly!
        verbose=False,
        use_TTA=False,  # Test time augmentation, False as default!
        save_format=None,  # ['nii','npy'], use 'nii' as default. Its purpose is for submission.
        snapshot=False,  # for visualization. Default false. It is recommended to generate the visualized figures.
        visual='',  # the path to save visualization
        postprocess=False,  # Default False, when use postprocess, the score of dice_ET would be changed.
        valid_in_train=False,  # if you are valid when train
        #grad_target=''
        ):
    
    from test import parser
    args = parser.parse_args()

    H, W, T = 240, 240, 160
    model.eval()

    runtimes = []
    ET_voxels_pred_list = []

    #edit - for dice metrics
    dice_NC_list = []
    dice_ET_list = []
    dice_ED_list = []
    IoU_NC_list = []
    IoU_ET_list = []
    IoU_ED_list = []


    for i, data in enumerate(valid_loader):
        print('-------------------------------------------------------------------')
        msg = 'Subject {}/{}, '.format(i + 1, len(valid_loader))
        if valid_in_train:
            data = [t.cuda(non_blocking=True) for t in data]
            x, target = data[:2]
        else:
            x = data
            x.cuda()

        if not use_TTA:
            torch.cuda.synchronize()  # add the code synchronize() to correctly count the runtime.
            start_time = time.time()
            logit = tailor_and_concat(x, model)

            torch.cuda.synchronize()
            elapsed_time = time.time() - start_time
            logging.info('Single sample test time consumption {:.2f} minutes!'.format(elapsed_time/60))
            runtimes.append(elapsed_time)


            if multimodel:
                logit = F.softmax(logit, dim=1)
                output = logit / 4.0

                load_file1 = load_file.replace('7998', '7996')
                if os.path.isfile(load_file1):
                    checkpoint = torch.load(load_file1)
                    model.load_state_dict(checkpoint['state_dict'])
                    print('Successfully load checkpoint {}'.format(load_file1))
                    logit = tailor_and_concat(x, model)
                    logit = F.softmax(logit, dim=1)
                    output += logit / 4.0
                load_file1 = load_file.replace('7998', '7997')
                if os.path.isfile(load_file1):
                    checkpoint = torch.load(load_file1)
                    model.load_state_dict(checkpoint['state_dict'])
                    print('Successfully load checkpoint {}'.format(load_file1))
                    logit = tailor_and_concat(x, model)
                    logit = F.softmax(logit, dim=1)
                    output += logit / 4.0
                load_file1 = load_file.replace('7998', '7999')
                if os.path.isfile(load_file1):
                    checkpoint = torch.load(load_file1)
                    model.load_state_dict(checkpoint['state_dict'])
                    print('Successfully load checkpoint {}'.format(load_file1))
                    logit = tailor_and_concat(x, model)
                    logit = F.softmax(logit, dim=1)
                    output += logit / 4.0
            else:
                output = F.softmax(logit, dim=1)

        else:
            x = x[..., :155]
            logit = F.softmax(tailor_and_concat(x, model), 1)  # no flip
            logit += F.softmax(tailor_and_concat(x.flip(dims=(2,)), model).flip(dims=(2,)), 1)  # flip H
            logit += F.softmax(tailor_and_concat(x.flip(dims=(3,)), model).flip(dims=(3,)), 1)  # flip W
            logit += F.softmax(tailor_and_concat(x.flip(dims=(4,)), model).flip(dims=(4,)), 1)  # flip D
            logit += F.softmax(tailor_and_concat(x.flip(dims=(2, 3)), model).flip(dims=(2, 3)), 1)  # flip H, W
            logit += F.softmax(tailor_and_concat(x.flip(dims=(2, 4)), model).flip(dims=(2, 4)), 1)  # flip H, D
            logit += F.softmax(tailor_and_concat(x.flip(dims=(3, 4)), model).flip(dims=(3, 4)), 1)  # flip W, D
            logit += F.softmax(tailor_and_concat(x.flip(dims=(2, 3, 4)), model).flip(dims=(2, 3, 4)), 1)  # flip H, W, D
            output = logit / 8.0  # mean

        # --- GENERATE GRAD-CAM ---
        # gradcam = GradCAM3D(model, target_layer_name='endconv', use_cuda=True)
        # if grad_target:
        #     # Ensure x has requires_grad=True for the backward pass
        #     x.requires_grad = True
            
        #     # Generate the 3D heatmap (D, H, W)
        #     cam_3d = gradcam.generate_cam(x, class_idx=grad_target)
            
        #     # Save the result
        #     name = names[i] if names else str(i)
        #     np.save(os.path.join(visual, f"{name}_gcam_cls{grad_target}.npy"), cam_3d)

        #Dice and IoU (requires labels)
        if valid_in_train:
            # output: (1, C, H, W, T) softmax probabilities (Tensor)
            # target: (1, H, W, T) labels (Tensor)

            with torch.no_grad():
                pred = output.argmax(dim=1)  # (1, H, W, T)

                # Necrotic Core (NC): label 1
                dice_NC = dice_score(
                    (pred == 1).float(),
                    (target == 1).float()
                )
                IoU_NC = mIOU(
                    (pred == 1).float(),
                    (target == 1).float()
                )

                # Enhancing Tumor (ET): label 2
                dice_ET = dice_score(
                    (pred == 2).float(),
                    (target == 2).float()
                )
                IoU_ET = mIOU(
                    (pred == 2).float(),
                    (target == 2).float()
                )

                # Edema (ED): label 3
                dice_ED = dice_score(
                    (pred == 3).float(),
                    (target == 3).float()
                )
                IoU_ED = mIOU(
                    (pred == 3).float(),
                    (target == 3).float()
                )

                dice_NC_list.append(dice_NC.item())
                dice_ET_list.append(dice_ET.item())
                dice_ED_list.append(dice_ED.item())
                IoU_NC_list.append(IoU_NC.item())
                IoU_ET_list.append(IoU_ET.item())
                IoU_ED_list.append(IoU_ED.item())

                print(
                    f'Dice | NC: {dice_NC:.4f}, '
                    f'ET: {dice_ET:.4f}, '
                    f'ED: {dice_ED:.4f}'
                )
                print(
                    f'IOU | NC: {IoU_NC:.4f}, '
                    f'ET: {IoU_ET:.4f}, '
                    f'ED: {IoU_ED:.4f}'
                )

        output = output[0, :, :H, :W, :T].cpu().detach().numpy()
        output = output.argmax(0)

        name = str(i)
        if names:
            name = names[i]
            msg += '{:>20}, '.format(name)

        print(msg)

        if savepath:
            # .npy for further model ensemble
            # .nii for directly model submission
            assert save_format in ['npy', 'nii']
            if save_format == 'npy':
                np.save(os.path.join(savepath, name + '_preds'), output)
            if save_format == 'nii':
                # raise NotImplementedError
                oname = os.path.join(savepath, name + '.nii.gz')
                seg_img = np.zeros(shape=(H, W, T), dtype=np.uint8)

                seg_img[np.where(output == 1)] = 1
                seg_img[np.where(output == 2)] = 2
                seg_img[np.where(output == 3)] = 3
                if verbose:
                    print('1:', np.sum(seg_img == 1), ' | 2:', np.sum(seg_img == 2), ' | 4:', np.sum(seg_img == 4))
                    print('WT:', np.sum((seg_img == 1) | (seg_img == 2) | (seg_img == 4)), ' | TC:',
                          np.sum((seg_img == 1) | (seg_img == 4)), ' | ET:', np.sum(seg_img == 4))
                nib.save(nib.Nifti1Image(seg_img, None), oname)
                print('Successfully save {}'.format(oname))

                if snapshot:
                    """ --- grey figure---"""
                    # Snapshot_img = np.zeros(shape=(H,W,T),dtype=np.uint8)
                    # Snapshot_img[np.where(output[1,:,:,:]==1)] = 64
                    # Snapshot_img[np.where(output[2,:,:,:]==1)] = 160
                    # Snapshot_img[np.where(output[3,:,:,:]==1)] = 255
                    """ --- colorful figure--- """
                    Snapshot_img = np.zeros(shape=(H, W, 3, T), dtype=np.uint8)
                    Snapshot_img[:, :, 0, :][np.where(output == 1)] = 255
                    Snapshot_img[:, :, 1, :][np.where(output == 2)] = 255
                    Snapshot_img[:, :, 2, :][np.where(output == 3)] = 255

                    for frame in range(T):
                        if not os.path.exists(os.path.join(visual, name)):
                            os.makedirs(os.path.join(visual, name))
                        # scipy.misc.imsave(os.path.join(visual, name, str(frame)+'.png'), Snapshot_img[:, :, :, frame])
                        imageio.imwrite(os.path.join(visual, name, str(frame)+'.png'), Snapshot_img[:, :, :, frame])

    mean_NC_dice = np.mean(dice_NC_list)
    mean_ET_dice = np.mean(dice_ET_list)
    mean_ED_dice = np.mean(dice_ED_list)
    mean_NC_IoU = np.mean(IoU_NC_list)
    mean_ET_IoU = np.mean(IoU_ET_list)
    mean_ED_IoU = np.mean(IoU_ED_list)


    generated_metrics = [[args.time_bucket, mean_NC_dice, mean_ET_dice, mean_ED_dice, mean_NC_IoU, mean_ET_IoU, mean_ED_IoU]]
    file = open(f"{args.modality_set}_{args.resolution}.csv", 'a', newline='')
    writer = csv.writer(file)
    writer.writerows(generated_metrics)
    file.close()

    if valid_in_train and len(dice_NC_list) > 0:
        print('----------------Final Dice----------------')
        print(f'Mean NC Dice: {mean_NC_dice:.4f}')
        print(f'Mean ET Dice: {mean_ET_dice:.4f}')
        print(f'Mean ED Dice: {mean_ED_dice:.4f}')
    if valid_in_train and len(dice_NC_list) > 0:
        print('----------------Final IOU----------------')
        print(f'Mean NC IOU: {mean_NC_IoU:.4f}')
        print(f'Mean ET IOU: {mean_ET_IoU:.4f}')
        print(f'Mean ED IOU: {mean_ED_IoU:.4f}')

    #print('runtimes:', sum(runtimes)/len(runtimes))