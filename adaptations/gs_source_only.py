import argparse
import glob
import logging
import math
import os
import os.path as osp
import sys
import time
from collections import OrderedDict

import torch.nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils import data
from torchvision import transforms

from adaptations.morph.FixedMorph import FixedMorph
from dataset.cs13_dataset_src import CS13SrcDataSet
from dataset.dp13_dataset import densepass13DataSetWeakStrong, densepass13TestDataSet
from model.stylegan_networks import StyleGAN2Discriminator
from model.trans4passplus import Trans4PASS_plus_v2
from morph.GaussianMorph import CONFIGS as CONFIGS_GM
from morph.GaussianMorph import GaussianMorph, Bilinear
from morph.losses import Grad
from utils.init import *
from utils.loss import BCEWithLogitsLossPixelWiseWeighted
from utils.transform import TensorFixScaleRandomCropWHBoth
from utils.val_util import Evaluator

INFO_DICT = {
    'CS19': {
        'RESTORE_FROM': '/home/jjiang/experiments/GCDDN/official_ckpt/trans4pass_plus_small_512x512.pth',
        'DATASET': CS13SrcDataSet,
        'DATA_DIRECTORY': '/home/jjiang/datasets/Cityscapes',
        'DATA_LIST_PATH': 'dataset/cityscapes_list/train.txt',
        'SOURCE_TRANSFORM': 'resize',
        'MORPH': {
            'LEARNING_RATE_MORPH': 1e-5,
            'LEARNING_RATE_D_MORPH': 1e-5,

            'GaussianMorph': {
                'loss_adv_target_rate': 10.0,
            },
            'TransMorph': {
                'loss_adv_target_rate': 20.0,
            },
            'FixedMorph': {

            },
            'NoMorph': {

            }
        }
    },
    'DP19': {
        'DATASET': densepass13DataSetWeakStrong,
        'DATASET_TEST': densepass13TestDataSet,
        'DATA_DIRECTORY_TARGET': '/home/jjiang/datasets/DensePASS/DensePASS',
        'DATA_LIST_PATH_TARGET': 'dataset/densepass_list/train.txt',
        'DATA_LIST_PATH_TARGET_TEST': 'dataset/densepass_list/val.txt',
        'INPUT_SIZE': '1024,512',
        'INPUT_SIZE_TARGET': '2048,400',
        'TARGET_TRANSFORM': 'FixScaleRandomCropWH',
        'INPUT_SIZE_TARGET_TEST': '2048,400',
        'NUM_CLASSES': 19,
        'NAME_CLASSES': [
            "road", "sidewalk", "building", "wall", "fence", "pole", "light", "sign", "vegetation",
            "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motocycle", "bicycle"
        ],
        'SAVE_PRED_EVERY': 250,
        'LEARNING_RATE': 5e-6
    },
}

# model special setting
MODEL = 'Source_only'
# model special setting done


IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

# model morph option
base_channel = 1  # 3 or 1
D_G_optimizer_name = 'RMSprop'  # RMSprop; Adam; SGD
MORPH_SCHEDULER_TYPE = 'poly'  # cos; poly
morph_weight_decay = 5e-5  # default 1e-5
# model morph option done

# model seg option
EMB_CHANS = 128
NUM_STEPS = 40000
WARM_UP_STEPS = int(NUM_STEPS * 0.2)  # warmup lr
NUM_STEPS_STOP = int(NUM_STEPS * 1.0)  # early stopping
SAVE_NUM_IMAGES = 2
SAVE_IMG_PRED_EVERY = 1000
SAVE_CKPT_EVERY = NUM_STEPS  # NUM_STEPS:no_save; 20000
POWER = 0.9

model_optimizer_name = 'SGD'  # SGD; AdamW
SCHEDULER_TYPE = 'poly'  # cos; poly
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
# model seg option done

# dataset and other option
BATCH_SIZE = 1
NUM_WORKERS = BATCH_SIZE * 4
IGNORE_LABEL = 255
RANDOM_SEED = 1234
SET = 'train'


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : Trans4PASS_v1, Trans4PASS_v2")
    parser.add_argument("--emb-chans", type=int, default=EMB_CHANS,
                        help="Number of channels in decoder head.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--num-classes", type=int, default=-1,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=-1,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default='',
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--cpu", action='store_true', help="choose to use cpu device.")
    parser.add_argument("--tensorboard", action='store_false', help="choose whether to use tensorboard.")
    parser.add_argument("--log-dir", type=str, default='',
                        help="Path to the directory of log.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    parser.add_argument("--morph-type", type=str, default='GaussianMorph',
                        choices=['GaussianMorph', 'NoMorph'],
                        help="continue training")
    parser.add_argument("--scene", default='outdoor', type=str, choices=['outdoor', 'indoor'])
    return parser.parse_args()


args = get_arguments()

# meta info
SCENE = args.scene
if SCENE == 'outdoor':
    SOURCE, TARGET = 'CS19', 'DP19'
else:
    raise Exception

# copy from info dict
LEARNING_RATE_D_MORPH = INFO_DICT[SOURCE]['MORPH']['LEARNING_RATE_D_MORPH']  # origin 1e-4
LEARNING_RATE_D_MORPH_MAX = LEARNING_RATE_D_MORPH * 2
LEARNING_RATE_MORPH = INFO_DICT[SOURCE]['MORPH']['LEARNING_RATE_MORPH']  # learning rate
LEARNING_RATE_MORPH_MAX = LEARNING_RATE_MORPH * 2

LEARNING_RATE = INFO_DICT[TARGET]['LEARNING_RATE']
args.learning_rate = LEARNING_RATE
LEARNING_RATE_MAX = LEARNING_RATE * 10

SAVE_PRED_EVERY = INFO_DICT[TARGET]['SAVE_PRED_EVERY']  # for example 10000->250
args.save_pred_every = SAVE_PRED_EVERY
NUM_CLASSES = INFO_DICT[TARGET]['NUM_CLASSES']
args.num_classes = NUM_CLASSES
NAME_CLASSES = INFO_DICT[TARGET]['NAME_CLASSES']

# morph info
MORPH = args.morph_type  # GaussianMorph, TransMorph, FixedMorph, NoMorph
MORPH_DICT = INFO_DICT[SOURCE]['MORPH'][MORPH]
MORPH_LOSS = True if MORPH in ['GaussianMorph'] else False


def setup_logger(name, save_dir, filename="log.txt", mode='w'):
    logging.root.name = name
    logging.root.setLevel(logging.INFO)
    # don't log results for the non-master process
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fh = logging.FileHandler(os.path.join(save_dir, filename), mode=mode)  # 'a+' for add, 'w' for overwrite
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logging.root.addHandler(fh)
    # else:
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logging.root.addHandler(ch)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def warmup_cosine_annealing_lr(current_step, total_steps, warmup_steps, base_lr, max_lr, end_lr):
    if current_step < warmup_steps:
        # Linear preheating stage
        warmup_lr = base_lr + (max_lr - base_lr) * current_step / warmup_steps
        return warmup_lr
    else:
        # Cosine annealing stage
        annealing_steps = total_steps - warmup_steps
        annealing_step = current_step - warmup_steps
        cosine_lr = end_lr + 0.5 * (max_lr - end_lr) * (1 + math.cos(math.pi * annealing_step / annealing_steps))
        return cosine_lr


def adjust_learning_rate(optimizer, i_iter):
    if SCHEDULER_TYPE == 'poly':
        lr = lr_poly(LEARNING_RATE, i_iter, args.num_steps, POWER)
    elif SCHEDULER_TYPE == 'cos':
        lr = warmup_cosine_annealing_lr(i_iter, args.num_steps, WARM_UP_STEPS, LEARNING_RATE, LEARNING_RATE_MAX,
                                        LEARNING_RATE)
    else:
        raise Exception
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
    return lr


def adjust_learning_rate_morph(optimizer, i_iter):
    if MORPH_SCHEDULER_TYPE == 'poly':
        lr = lr_poly(LEARNING_RATE_MORPH, i_iter, args.num_steps, POWER)
    elif MORPH_SCHEDULER_TYPE == 'cos':
        lr = warmup_cosine_annealing_lr(i_iter, args.num_steps, WARM_UP_STEPS, LEARNING_RATE_MORPH,
                                        LEARNING_RATE_MORPH_MAX,
                                        LEARNING_RATE_MORPH)
    else:
        raise Exception
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
    return lr


def adjust_learning_rate_D_morph(optimizer, i_iter):
    if MORPH_SCHEDULER_TYPE == 'poly':
        lr = lr_poly(LEARNING_RATE_D_MORPH, i_iter, args.num_steps, POWER)
    elif MORPH_SCHEDULER_TYPE == 'cos':
        lr = warmup_cosine_annealing_lr(i_iter, args.num_steps, WARM_UP_STEPS, LEARNING_RATE_D_MORPH,
                                        LEARNING_RATE_D_MORPH_MAX,
                                        LEARNING_RATE_D_MORPH)
    else:
        raise Exception
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
    return lr


@torch.no_grad()
def _update_teacher_model(student_model, teacher_model, keep_rate=0.9996):
    student_model_dict = student_model.state_dict()

    new_teacher_dict = OrderedDict()
    for key, value in teacher_model.state_dict().items():
        if key in student_model_dict.keys():
            new_teacher_dict[key] = (
                    student_model_dict[key] *
                    (1 - keep_rate) + value * keep_rate
            )
        else:
            raise Exception("{} is not found in student model".format(key))

    teacher_model.load_state_dict(new_teacher_dict)


def main():
    """Create the model and start the training."""
    # set random seed
    set_random_seed(args.random_seed)

    # set para
    input_nc = base_channel

    # change args
    DIR_NAME = '{}_{}_{}_{}_'.format(MODEL, MORPH, SOURCE, TARGET)
    SNAPSHOT_DIR = 'snapshots/' + DIR_NAME
    LOG_DIR = SNAPSHOT_DIR
    exp_name = args.snapshot_dir
    args.snapshot_dir = SNAPSHOT_DIR + exp_name
    args.log_dir = LOG_DIR + exp_name
    TIME_STAMP = time.strftime('%Y-%m-%d-%H-%M', time.localtime())
    setup_logger('Trans4PASS', args.log_dir, f'{TIME_STAMP}_log.txt')

    os.makedirs(args.snapshot_dir, exist_ok=True)

    device = torch.device("cuda" if not args.cpu else "cpu")

    w, h = map(int, INFO_DICT[TARGET]['INPUT_SIZE'].split(','))
    input_size = (w, h)

    w, h = map(int, INFO_DICT[TARGET]['INPUT_SIZE_TARGET'].split(','))
    input_size_target = (w, h)

    w, h = map(int, INFO_DICT[TARGET]['INPUT_SIZE_TARGET_TEST'].split(','))
    input_size_target_test = (w, h)

    Iter = 0
    bestIoU = 0
    mIoU = 0
    mIoU_teacher = 0

    norm = transforms.Normalize((.485, .456, .406), (.229, .224, .225))
    freeze_model(norm)
    # Create network
    # init G
    model = Trans4PASS_plus_v2(num_classes=args.num_classes, emb_chans=args.emb_chans, norm=norm)

    saved_state_dict = torch.load(INFO_DICT[SOURCE]['RESTORE_FROM'], map_location=lambda storage, loc: storage)
    if 'state_dict' in saved_state_dict.keys():
        saved_state_dict = saved_state_dict['state_dict']
    msg = model.load_state_dict(saved_state_dict, strict=False)
    logging.info(msg)

    unfreeze_model(model)
    model.to(device)

    # init optimizer
    if model_optimizer_name == 'SGD':
        optimizer_seg = optim.SGD(model.parameters(),
                                  lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif model_optimizer_name == 'AdamW':
        optimizer_seg = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise Exception
    optimizer_seg.zero_grad()

    # get deformable fild
    '''
    Initialize model
    '''
    # smooth loss
    recon_loss = torch.nn.MSELoss(reduction='mean')  # torch.nn.L1Loss(); torch.nn.MSELoss()
    sem_loss = torch.nn.MSELoss(reduction='mean')  # torch.nn.L1Loss(); torch.nn.MSELoss()

    if MORPH == 'GaussianMorph':
        config = CONFIGS_GM['GaussianMorph']
        smooth_fuc = Grad(penalty='l2', reduction='sum')  # penalty='l1'; penalty='l2'
        model_morph = GaussianMorph(config, smooth_fuc=smooth_fuc, device=device, channel=base_channel,
                                    input_size=(input_size[1], input_size[0]))
    else:
        model_morph = FixedMorph(input_size=input_size, device=device)

    # init D_MORPH
    loss_smooth_rate = 1.0
    loss_recon_rate = loss_smooth_rate
    loss_sem_rate = loss_smooth_rate

    D_G_name = 'StyleGAN2Discriminator'  # StyleGAN2Discriminator

    if MORPH_LOSS:
        model_morph.to(device)
        unfreeze_model(model_morph)

        optimizer_morph = optim.Adam(model_morph.parameters(), lr=LEARNING_RATE_MORPH, weight_decay=morph_weight_decay,
                                     amsgrad=True)
        optimizer_morph.zero_grad()

        loss_adv_target_rate = MORPH_DICT['loss_adv_target_rate']

        if D_G_name == 'StyleGAN2Discriminator':
            model_D_morph_pin2pan = StyleGAN2Discriminator(input_nc=input_nc, size=1024).to(device)
        else:
            raise Exception
        unfreeze_model(model_D_morph_pin2pan)
        model_D_morph_pin2pan.to(device)

        if D_G_optimizer_name == 'Adam':
            optimizer_D_morph_pin2pan = optim.Adam(model_D_morph_pin2pan.parameters(), lr=LEARNING_RATE_D_MORPH,
                                                   betas=(0.9, 0.99),
                                                   weight_decay=morph_weight_decay)
        elif D_G_optimizer_name == 'SGD':
            optimizer_D_morph_pin2pan = optim.SGD(model_D_morph_pin2pan.parameters(), lr=LEARNING_RATE_D_MORPH,
                                                  momentum=args.momentum,
                                                  weight_decay=morph_weight_decay)
        elif D_G_optimizer_name == 'RMSprop':
            optimizer_D_morph_pin2pan = optim.RMSprop(model_D_morph_pin2pan.parameters(),
                                                      lr=LEARNING_RATE_D_MORPH, momentum=args.momentum,
                                                      weight_decay=morph_weight_decay)
        else:
            raise Exception
        optimizer_D_morph_pin2pan.zero_grad()

    # init data loader
    trainset_pin = INFO_DICT[SOURCE]['DATASET'](INFO_DICT[SOURCE]['DATA_DIRECTORY'],
                                                INFO_DICT[SOURCE]['DATA_LIST_PATH'],
                                                max_iters=args.num_steps * BATCH_SIZE,
                                                crop_size=input_size, scale=args.random_scale,
                                                mirror=args.random_mirror,
                                                mean=IMG_MEAN, set=args.set, need_grey=base_channel == 1,
                                                normalize=False,
                                                trans=INFO_DICT[SOURCE]['SOURCE_TRANSFORM'], org_mapping=True)

    trainloader_pin = data.DataLoader(trainset_pin, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers,
                                      pin_memory=True)
    trainloader_iter_pin = enumerate(trainloader_pin)

    strong_aug_list = ['color jittering', 'Gaussian blur', 'cutout patches']
    targetset = INFO_DICT[TARGET]['DATASET'](INFO_DICT[TARGET]['DATA_DIRECTORY_TARGET'],
                                             INFO_DICT[TARGET]['DATA_LIST_PATH_TARGET'],
                                             max_iters=args.num_steps * BATCH_SIZE,
                                             crop_size=input_size_target, scale=False, mirror=args.random_mirror,
                                             mean=IMG_MEAN,
                                             set=args.set,
                                             trans=INFO_DICT[TARGET]['TARGET_TRANSFORM'], need_grey=base_channel == 1,
                                             normalize=False,
                                             strong_aug_list=strong_aug_list, org_mapping=True)

    targetloader = data.DataLoader(targetset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                   pin_memory=True)
    targetloader_iter = enumerate(targetloader)

    logging.info('\n--- load TEST dataset ---')

    # test_h, test_w = 400, 2048
    test_w, test_h = input_size_target_test
    targettestset = INFO_DICT[TARGET]['DATASET_TEST'](INFO_DICT[TARGET]['DATA_DIRECTORY_TARGET'],
                                                      INFO_DICT[TARGET]['DATA_LIST_PATH_TARGET_TEST'],
                                                      crop_size=(test_w, test_h),
                                                      mean=IMG_MEAN, scale=False, mirror=False, set='val',
                                                      normalize=False, org_mapping=True)
    testloader = data.DataLoader(targettestset, batch_size=1, shuffle=False, pin_memory=True)

    # init loss
    bce_loss = torch.nn.BCEWithLogitsLoss()
    bce_loss_with_weight = BCEWithLogitsLossPixelWiseWeighted()
    seg_loss = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)

    # labels for adversarial training
    source_label = 0
    target_label = 1
    eps = 0.1
    smoothed_source_label = source_label + eps
    smoothed_target_label = target_label - eps

    # set up tensor board
    if args.tensorboard:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        writer = SummaryWriter(args.log_dir)

    '''
    Initialize spatial transformation function
    '''
    reg_model = Bilinear(zero_boundary=True, mode='nearest').cuda()
    freeze_model(reg_model)
    reg_model_bilin = Bilinear(zero_boundary=True, mode='bilinear').cuda()
    freeze_model(reg_model_bilin)

    grid_img = mk_grid_img(32, 1, (args.batch_size, input_size[1], input_size[0]))

    # draw
    plt.switch_backend('agg')
    evaluator = Evaluator(testloader, NUM_CLASSES, NAME_CLASSES, device, softmax=False)
    # start training
    for i_iter in range(Iter, args.num_steps):
        # reset optimizer
        optimizer_seg.zero_grad()
        lr_model = adjust_learning_rate(optimizer_seg, i_iter)

        # model freeze & unfreeze
        unfreeze_model(model)
        if MORPH_LOSS:
            optimizer_morph.zero_grad()
            lr_morph = adjust_learning_rate_morph(optimizer_morph, i_iter)

            optimizer_D_morph_pin2pan.zero_grad()
            lr_D_morph = adjust_learning_rate_D_morph(optimizer_D_morph_pin2pan, i_iter)

            unfreeze_model(model_morph)
            unfreeze_model(model_D_morph_pin2pan)
        else:
            lr_morph = 0.
            lr_D_morph = 0.
            loss_smooth = torch.zeros([1]).to(device)
            loss_recon = torch.zeros([1]).to(device)
            loss_sem = torch.zeros([1]).to(device)
            loss_adv_target_all = torch.zeros([1]).to(device)
            loss_D_value_morph = 0.

        # get data
        _, batch_source_pin = trainloader_iter_pin.__next__()
        images_source_pin_origin, images_source_pin_grey, labels_source_pin, _, _ = batch_source_pin
        images_source_pin_origin = images_source_pin_origin.to(device)
        images_source_pin_grey = images_source_pin_grey.to(device)
        labels_source_pin = labels_source_pin.long().to(device)

        _, batch_target = targetloader_iter.__next__()
        if SCENE == 'outdoor':
            images_target, images_target_strong, _, _, images_target_grey, _, _ = batch_target
        else:
            raise Exception
        images_target = images_target.to(device)
        images_target_strong = images_target_strong.to(device)
        images_target_grey = images_target_grey.to(device)

        if SCENE == 'outdoor':
            crop_opt = TensorFixScaleRandomCropWHBoth(input_size_target, input_size)
            images_target_cropped = crop_opt(images_target)
        else:
            raise Exception

        # color convert
        images_source_pin = images_source_pin_origin

        # prepare done
        # warp with flow
        if MORPH == 'GaussianMorph':
            images_pin2pan_grey, flow_pin2pan, images_pan2pin_grey, flow_pan2pin, log_info, level_grids = model_morph(
                (images_source_pin_grey, images_target_grey))
            loss_smooth = model_morph.scale_reg_loss()
        elif MORPH == 'NoMorph':
            flow_pin2pan = model_morph.affine_grid
            flow_pan2pin = model_morph.affine_grid_inverse
        else:
            raise Exception

        # get images_source2target and images_target2source
        images_pin2pan = reg_model_bilin(images_source_pin.float(), flow_pin2pan)
        images_pan2pin = reg_model_bilin(images_target_cropped.float(), flow_pan2pin)

        # deal with label 0, due to reg_model may generate black edge, which value is also 0
        with torch.no_grad():
            labels_pin2pan = warp_labels(labels_source_pin, flow_pin2pan, reg_model, ignore_label=IGNORE_LABEL)
            labels_pin2pan = labels_pin2pan.detach()
        # warp with flow done

        # ---step1 train model_seg---
        # 1. L^seg
        _, pred_pin2pan = model(images_pin2pan.detach())
        # 2. L^col_S
        loss_pin = seg_loss(pred_pin2pan, labels_pin2pan)

        # total loss
        model_seg_total_loss = loss_pin
        model_seg_total_loss.backward()

        optimizer_seg.step()
        # ---step1 train model_seg done---

        if MORPH_LOSS:
            # ---step2 train model_morph---
            # freeze model
            freeze_model(model)
            # train G
            freeze_model(model_D_morph_pin2pan)

            # warp with flow
            images_pin_reconstruct = reg_model_bilin(images_pin2pan, flow_pan2pin)
            images_pan_reconstruct = reg_model_bilin(images_pan2pin, flow_pin2pan)
            loss_recon = recon_loss(images_source_pin, images_pin_reconstruct) \
                         + recon_loss(images_target_cropped, images_pan_reconstruct)
            loss_recon = loss_recon * loss_recon_rate

            loss_sem = sem_loss(model(images_pin2pan)[-1].float(),
                                reg_model_bilin(model(images_source_pin)[-1], flow_pin2pan)) + \
                       sem_loss(model(images_pan2pin)[-1].float(),
                                reg_model_bilin(model(images_target_cropped)[-1], flow_pan2pin))

            loss_sem = loss_sem * loss_sem_rate

            # D
            loss_adv_target_all = 0.
            if D_G_name == 'StyleGAN2Discriminator':
                D_out = model_D_morph_pin2pan(images_pin2pan_grey)
                loss_adv_target = bce_loss(D_out,
                                           torch.FloatTensor(D_out.data.size()).fill_(smoothed_target_label).to(device))
                loss_adv_target_all += loss_adv_target * loss_adv_target_rate
            else:
                raise Exception

            # smooth
            loss_smooth = loss_smooth * loss_smooth_rate

            loss_morph = (i_iter / args.num_steps) * (loss_recon + loss_sem) + loss_smooth + loss_adv_target_all
            loss_morph.backward()

            # === train D G
            unfreeze_model(model_D_morph_pin2pan)

            # train with source
            loss_D_value_morph = 0.
            images_pin2pan_grey = images_pin2pan_grey.detach()
            images_target_grey = images_target_grey.detach()

            if D_G_name == 'StyleGAN2Discriminator':
                D_out = model_D_morph_pin2pan(images_pin2pan_grey)

                loss_D = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).to(device))
                loss_D = loss_D / 2
                loss_D.backward()
                loss_D_value_morph += loss_D.item()

                # train with target
                D_out = model_D_morph_pin2pan(images_target_grey)

                loss_D = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(smoothed_target_label).to(device))
                loss_D = loss_D / 2
                loss_D.backward()
                loss_D_value_morph += loss_D.item()
            else:
                raise Exception

            optimizer_morph.step()
            optimizer_D_morph_pin2pan.step()
        # ---step2 train model_morph done---
        scalar_info = {
            'model_seg_total_loss': model_seg_total_loss.item(),
            'loss_pin': loss_pin.item(),

            'loss_recon': loss_recon.item(),
            'loss_sem': loss_sem.item(),
            'loss_smooth': loss_smooth.item(),
            'loss_adv_target_all': loss_adv_target_all.item(),
            'loss_D_value_morph': loss_D_value_morph,
        }
        if MORPH == 'GaussianMorph':
            scalar_info.update(log_info)

        if args.tensorboard:
            if i_iter % 10 == 0:
                for key, val in scalar_info.items():
                    writer.add_scalar(key, val, i_iter)
                writer.add_scalar('lr/lr_model', lr_model, i_iter)
                writer.add_scalar('lr/lr_morph', lr_morph, i_iter)
                writer.add_scalar('lr/lr_D_morph', lr_D_morph, i_iter)

                writer.add_scalar('miou/mIoU', mIoU, i_iter)
                writer.add_scalar('miou/mIoU_teacher', mIoU_teacher, i_iter)

            if i_iter % SAVE_IMG_PRED_EVERY == 0:
                with torch.no_grad():
                    # draw
                    # 1. draw grid
                    logging.info('drawing visualization figures!')
                    grid_pin2pan = reg_model(grid_img.float(), flow_pin2pan)
                    grid_pin2pan_image = comput_fig(grid_pin2pan)
                    writer.add_figure('grid_pin2pan_image', grid_pin2pan_image, i_iter)
                    plt.close(grid_pin2pan_image)

                    grid_pan2pin = reg_model(grid_img.float(), flow_pan2pin)
                    grid_pan2pin_image = comput_fig(grid_pan2pin)
                    writer.add_figure('grid_pan2pin_image', grid_pan2pin_image, i_iter)
                    plt.close(grid_pan2pin_image)

                    # 2. draw image
                    # draw source pin
                    images_source_pin_origin_figure = comput_fig(images_source_pin_origin)
                    writer.add_figure('images_source_pin_origin_figure', images_source_pin_origin_figure, i_iter)
                    plt.close(images_source_pin_origin_figure)

                    images_source_pin_figure = comput_fig(images_source_pin)
                    writer.add_figure('images_source_pin_figure', images_source_pin_figure, i_iter)
                    plt.close(images_source_pin_figure)

                    images_pin2pan_figure = comput_fig(images_pin2pan)
                    writer.add_figure('images_pin2pan_figure', images_pin2pan_figure, i_iter)
                    plt.close(images_pin2pan_figure)

                    images_pin_with_grid = draw_with_grid(images_source_pin, grid_img)
                    images_pin2pan_with_grid = reg_model(images_pin_with_grid, flow_pin2pan)
                    images_pin2pan_with_grid_figure = comput_fig(images_pin2pan_with_grid)
                    writer.add_figure('images_pin2pan_with_grid_figure', images_pin2pan_with_grid_figure,
                                      i_iter)
                    plt.close(images_pin2pan_with_grid_figure)

                    images_pin_reconstruct_with_grid = reg_model(images_pin2pan_with_grid, flow_pan2pin)
                    images_pin_reconstruct_with_grid_figure = comput_fig(images_pin_reconstruct_with_grid)
                    writer.add_figure('images_pin_reconstruct_with_grid_figure',
                                      images_pin_reconstruct_with_grid_figure, i_iter)
                    plt.close(images_pin_reconstruct_with_grid_figure)

                    # draw source pan
                    images_target_cropped_figure = comput_fig(images_target_cropped)
                    writer.add_figure('images_target_cropped_figure', images_target_cropped_figure, i_iter)
                    plt.close(images_target_cropped_figure)

                    images_pan_with_grid = draw_with_grid(images_target_cropped, grid_img)
                    images_pan2pin_with_grid = reg_model(images_pan_with_grid, flow_pan2pin)
                    images_pan2pin_with_grid_figure = comput_fig(images_pan2pin_with_grid)
                    writer.add_figure('images_pan2pin_with_grid_figure', images_pan2pin_with_grid_figure,
                                      i_iter)
                    plt.close(images_pan2pin_with_grid_figure)

                    # [option] closer look at target labels
                    images_target_figure = comput_fig(images_target)
                    writer.add_figure('images_target_figure', images_target_figure, i_iter)
                    plt.close(images_target_figure)

                    images_target_strong_figure = comput_fig(images_target_strong)
                    writer.add_figure('images_target_strong_figure', images_target_strong_figure, i_iter)
                    plt.close(images_target_strong_figure)

                    if MORPH == 'GaussianMorph':
                        # level grid
                        for i, level_grid in enumerate(level_grids):
                            level_grid_img = reg_model(grid_img.float(), level_grid)
                            level_grid_img = comput_fig(level_grid_img)
                            writer.add_figure(f'level_{i}_grid_img', level_grid_img, i_iter)
                            plt.close(level_grid_img)

                    draw_option = False
                    if draw_option:
                        # background ignore is white
                        # [option] closer look at source labels
                        labels_pin_figure = comput_fig(get_colored_labels(labels_source_pin))
                        writer.add_figure('labels_pin_figure', labels_pin_figure, i_iter)
                        plt.close(labels_pin_figure)
                        # [option] closer look at source2target labels
                        labels_pin2pan_figure = comput_fig(get_colored_labels(labels_pin2pan))
                        writer.add_figure('labels_pin2pan_figure', labels_pin2pan_figure, i_iter)
                        plt.close(labels_pin2pan_figure)

                    logging.info('draw visualization figures done!')

        if i_iter % 10 == 0:
            logging.info('iter = {0:8d}/{1:8d}, losses:{2}'.format(
                i_iter, args.num_steps, scalar_info))

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            logging.info('taking snapshot ...')

            freeze_model(model)
            eval_result = evaluator([model], i_iter, [1.0])
            unfreeze_model(model)

            mIoU = eval_result['mIoUs'][0]
            best_miou_str = eval_result['best_miou_str']
            logging.info(best_miou_str)

            if mIoU >= bestIoU:
                bestIoU = mIoU
                pre_filename = osp.join(args.snapshot_dir + 'best*.pth')
                pre_filename = glob.glob(pre_filename)
                try:
                    for p in pre_filename:
                        os.remove(p)
                except OSError as e:
                    logging.info(e)
                torch.save(model.state_dict(),
                           osp.join(args.snapshot_dir, 'best.pth'))
                if MORPH_LOSS:
                    torch.save(model_morph.state_dict(),
                               osp.join(args.snapshot_dir, 'best_morph.pth'))
                    torch.save(model_D_morph_pin2pan.state_dict(),
                               osp.join(args.snapshot_dir, 'best_D_morph_pin2pan.pth'))
                with open(osp.join(args.snapshot_dir, f'{TIME_STAMP}_best_miou.txt'), mode='w', encoding='utf-8') as f:
                    f.write(best_miou_str)

        if i_iter >= args.num_steps_stop - 1:
            break

        if i_iter != 0 and i_iter % SAVE_CKPT_EVERY == 0:
            logging.info('save model ...')
            torch.save(model.state_dict(),
                       osp.join(args.snapshot_dir, f'iter{i_iter}.pth'))
            if MORPH_LOSS:
                torch.save(model_morph.state_dict(),
                           osp.join(args.snapshot_dir, f'iter{i_iter}_morph.pth'))
                torch.save(model_D_morph_pin2pan.state_dict(),
                           osp.join(args.snapshot_dir, f'iter{i_iter}_D_morph_pin2pan.pth'))

    # save the last one
    logging.info('save model ...')
    torch.save(model.state_dict(),
               osp.join(args.snapshot_dir, 'latest.pth'))
    if MORPH_LOSS:
        torch.save(model_morph.state_dict(),
                   osp.join(args.snapshot_dir, 'latest_morph.pth'))
        torch.save(model_D_morph_pin2pan.state_dict(),
                   osp.join(args.snapshot_dir, 'latest_D_morph_pin2pan.pth'))

    if args.tensorboard:
        writer.close()


if __name__ == '__main__':
    main()
