import argparse
import os
import time
import random
import numpy as np
import setproctitle

import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torch.optim
from torch.utils.data import DataLoader

from data.BraTS import BraTS
from predict import validate_softmax
from models.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS

TIME_BUCKETS = {
    'all':      [0, 100000],
    "0":        [0,0],
    "1-12":     [1,12],
    "13-24":    [13, 24],
    "25-36":    [25, 36],
    "37-48":    [37, 48],
    "49-60":    [49, 60],
    "61-72":    [61, 72],
    "73-84":    [73, 84],
    "85-96":    [85, 96],
    "97-108":   [97, 108],
    "109-120":  [109, 120],
    "121-132":  [121, 132],
    "133-144":  [133, 144],
    "145-156":  [145, 156],
    "157-168":  [157, 168],
    "169-180":  [169, 180],
    "181-192":  [181, 192],
    "193-204":  [193, 204],
    "205-216":  [205, 216],
    "217-228":  [217, 228],
    "229-242":  [229, 242],
    "243+":     [243, 100000]
    }

local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

parser = argparse.ArgumentParser()

parser.add_argument('--user', default='team kams', type=str)

parser.add_argument('--root', default='/lambda/nfs/KAMS/', type=str)

parser.add_argument('--valid_dir', default='TransBTS', type=str)

parser.add_argument('--valid_file', default='valid.txt', type=str)

parser.add_argument('--data_dir', default='Imaging', type=str)

parser.add_argument('--output_dir', default='output', type=str)

parser.add_argument('--submission', default='submission', type=str)

parser.add_argument('--visual', default='visualization', type=str)

parser.add_argument('--experiment', default='TransBTS', type=str)

parser.add_argument('--test_date', default=local_time.split(' ')[0], type=str) #default current date

parser.add_argument('--test_file', default='model_epoch_last.pth', type=str)

parser.add_argument('--use_TTA', default=True, type=bool)

parser.add_argument('--post_process', default=True, type=bool)

parser.add_argument('--save_format', default='nii', choices=['npy', 'nii'], type=str)

parser.add_argument('--crop_H', default=128, type=int)

parser.add_argument('--crop_W', default=128, type=int)

parser.add_argument('--crop_D', default=128, type=int)

parser.add_argument('--seed', default=1000, type=int)

parser.add_argument('--model_name', default='TransBTS', type=str)

parser.add_argument('--num_class', default=4, type=int)

parser.add_argument('--no_cuda', default=False, type=bool)

parser.add_argument('--gpu', default='0', type=str)

parser.add_argument('--num_workers', default=4, type=int)

parser.add_argument('--modality_set', default='all', type=str,
                    choices=['flair', 'ct1', 't1', 't2', 'ct1_flair', 't1_t2', 'all'])

parser.add_argument('--resolution', default=1.0, type=float, choices=[1.0, 0.75, 0.5])

parser.add_argument('--input_C', default=4, type=int) #Set as 1 (only one modalitiy), 2 (two modality pairs), or 4 (all four modalities)

parser.add_argument('--version', default='1', type=str) 

parser.add_argument('--time_bucket', default='', type=str, choices=['all','0','1-12','13-24','25-36','37-48','49-60','61-72','73-84','85-96','97-108','109-120','121-132','133-144','145-156','157-168','169-180','181-192','193-204','205-216','217-228','229-242','243+'])

args = parser.parse_args()


def main():

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    _, model = TransBTS(dataset='brats', _conv_repr=True, _pe_type="learned", input_channels=args.input_C)

    model = torch.nn.DataParallel(model).cuda()

    load_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'checkpoint', args.modality_set+'-{}'.format(args.resolution)+'-v{}'.format(args.version), args.test_file)

    if os.path.exists(load_file):
        checkpoint = torch.load(load_file, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        args.start_epoch = checkpoint['epoch']
        print('Successfully load checkpoint {}'.format(os.path.join(args.modality_set+'-{}'.format(args.resolution)+'-v{}'.format(args.version), args.test_file)))
    else:
        print('There is no resume file to load!')

    valid_txt = os.path.join(args.root, args.valid_dir, args.valid_file)

    with open(valid_txt, 'r', encoding='utf-8') as f_in, \
        open('valid_list', 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            parts = line.split(os.sep)            
            if len(parts) > 1:
                time_bucket = int(parts[1][5:8])
                
                if TIME_BUCKETS[args.time_bucket][0] <= time_bucket <= TIME_BUCKETS[args.time_bucket][1]:
                    f_out.write(line)

    print(f"File 'valid_list' created successfully.")

    valid_list = os.path.join(args.root, args.valid_dir, 'valid_list')
    
    valid_root = os.path.join(args.root, args.data_dir)
    valid_set = BraTS(valid_list, valid_root, mode='valid', modality_set = args.modality_set)
    print('Samples for valid = {}'.format(len(valid_set)))

    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    submission = os.path.join(os.path.abspath(os.path.dirname(__file__)), args.output_dir,
                              args.submission, args.modality_set+'-{}'.format(args.resolution)+'-v{}'.format(args.version))
    visual = os.path.join(os.path.abspath(os.path.dirname(__file__)), args.output_dir,
                          args.visual, args.modality_set+'-{}'.format(args.resolution)+'-v{}'.format(args.version))

    if not os.path.exists(submission):
        os.makedirs(submission)
    if not os.path.exists(visual):
        os.makedirs(visual)

    start_time = time.time()

    names = []
    with open('valid_list') as f:
        for line in f:
            line = line.strip()
            parts = line.split(os.sep)
            name = parts[0] + '_' + parts[1]
            names.append(name)

    with torch.no_grad():
        validate_softmax(valid_loader=valid_loader,
                         model=model,
                         load_file=load_file,
                         multimodel=False,
                         savepath=submission,
                         visual=visual,
                         names=names,
                         use_TTA=args.use_TTA,
                         save_format=args.save_format,
                         snapshot=True,
                         postprocess=True,
                         valid_in_train=True
                         )

    end_time = time.time()
    full_test_time = (end_time-start_time)/60
    average_time = full_test_time/len(valid_set)
    print('{:.2f} minutes!'.format(average_time))


if __name__ == '__main__':
    # config = opts()
    setproctitle.setproctitle('{}: Testing!'.format(args.user))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    main()


