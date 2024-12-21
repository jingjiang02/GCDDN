import copy
import datetime
import os
import sys
import time

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import logging
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.nn import CrossEntropyLoss

from torchvision import transforms
from segmentron.data.dataloader import get_segmentation_dataset
from segmentron.models.model_zoo import get_segmentation_model
from segmentron.solver.optimizer import get_optimizer
from segmentron.solver.lr_scheduler import get_scheduler
from segmentron.utils.distributed import *
from segmentron.utils.score import SegmentationMetric
from segmentron.utils.filesystem import save_checkpoint
from segmentron.utils.options import parse_args
from segmentron.utils.default_setup import default_setup
from segmentron.utils.visualize import show_flops_params
from segmentron.config import cfg
from tabulate import tabulate

try:
    import apex
except:
    print('apex is not installed')


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        # dataset and dataloader
        data_kwargs = {'transform': input_transform,
                       'base_size': cfg.TRAIN.BASE_SIZE,
                       'crop_size': cfg.TRAIN.CROP_SIZE}

        data_kwargs_testval = {'transform': input_transform,
                               'base_size': cfg.TRAIN.BASE_SIZE,
                               'crop_size': cfg.TEST.CROP_SIZE}

        train_dataset = get_segmentation_dataset(cfg.DATASET.NAME, split='train', mode='train', **data_kwargs)
        # --- monitor
        test_dataset_2 = get_segmentation_dataset('densepass', split='val', mode='val', **data_kwargs_testval)

        self.classes = train_dataset.classes

        self.iters_per_epoch = len(train_dataset) // (args.num_gpus * cfg.TRAIN.BATCH_SIZE)
        self.max_iters = cfg.TRAIN.EPOCHS * self.iters_per_epoch

        train_sampler = make_data_sampler(train_dataset, shuffle=True, distributed=args.distributed)
        train_batch_sampler = make_batch_data_sampler(train_sampler, cfg.TRAIN.BATCH_SIZE, self.max_iters,
                                                      drop_last=True)

        # --- monitor
        test_sampler_2 = make_data_sampler(test_dataset_2, False, args.distributed)
        test_batch_sampler_2 = make_batch_data_sampler(test_sampler_2, 1, drop_last=False)

        self.train_loader = data.DataLoader(dataset=train_dataset,
                                            batch_sampler=train_batch_sampler,
                                            num_workers=cfg.DATASET.WORKERS,
                                            pin_memory=True)
        # --- monitor
        self.test_loader_2 = data.DataLoader(dataset=test_dataset_2,
                                             batch_sampler=test_batch_sampler_2,
                                             num_workers=cfg.DATASET.WORKERS,
                                             pin_memory=True)

        # create network
        self.model = get_segmentation_model().to(self.device)

        # print params and flops
        if get_rank() == 0:
            try:
                show_flops_params(copy.deepcopy(self.model), args.device)
            except Exception as e:
                logging.warning('get flops and params error: {}'.format(e))
        if cfg.MODEL.BN_TYPE not in ['BN']:
            logging.info('Batch norm type is {}, convert_sync_batchnorm is not effective'.format(cfg.MODEL.BN_TYPE))
        elif args.distributed and cfg.TRAIN.SYNC_BATCH_NORM:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            logging.info('SyncBatchNorm is effective!')
        else:
            logging.info('Not use SyncBatchNorm!')
        # create criterion
        self.criterion = CrossEntropyLoss(ignore_index=cfg.DATASET.IGNORE_INDEX).to(self.device)
        self.optimizer = get_optimizer(self.model)

        # lr scheduling
        self.lr_scheduler = get_scheduler(self.optimizer, max_iters=self.max_iters,
                                          iters_per_epoch=self.iters_per_epoch)
        # resume checkpoint if needed
        self.start_epoch = 0
        if args.resume and os.path.isfile(args.resume):
            name, ext = os.path.splitext(args.resume)
            assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
            logging.info('Resuming training, loading {}...'.format(args.resume))
            resume_sate = torch.load(args.resume, map_location=args.device)
            self.model.load_state_dict(resume_sate['state_dict'])
            self.start_epoch = resume_sate['epoch']
            logging.info('resume train from epoch: {}'.format(self.start_epoch))
            if resume_sate['optimizer'] is not None and resume_sate['lr_scheduler'] is not None:
                logging.info('resume optimizer and lr scheduler from resume state..')
                self.optimizer.load_state_dict(resume_sate['optimizer'])
                self.lr_scheduler.load_state_dict(resume_sate['lr_scheduler'])

        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model,
                                                             device_ids=[args.local_rank],
                                                             output_device=args.local_rank,
                                                             find_unused_parameters=False)
        # evaluation metrics
        self.metric = SegmentationMetric(train_dataset.num_class, args.distributed)
        # --- monitor
        self.best_test_2_mIoU = 0.
        self.cur_test_2_mIoU = 0.

        self.normalize = transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD)

    def train(self):
        self.save_to_disk = get_rank() == 0
        epochs, max_iters, iters_per_epoch = cfg.TRAIN.EPOCHS, self.max_iters, self.iters_per_epoch
        log_per_iters, val_per_iters = self.args.log_iter, self.args.val_epoch * self.iters_per_epoch

        start_time = time.time()
        logging.info('Start training, Total Epochs: {:d} = Total Iterations {:d}'.format(epochs, max_iters))

        self.model.train()
        iteration = self.start_epoch * iters_per_epoch if self.start_epoch > 0 else 0
        train_iters = iter(self.train_loader)
        for _ in range(len(self.train_loader)):
            epoch = iteration // iters_per_epoch + 1
            iteration += 1

            # get data
            images_source, targets, _ = next(train_iters)
            images_source = images_source.to(self.device)
            targets = targets.to(self.device)

            _, output_pred = self.model(self.normalize(images_source))
            losses = self.criterion(output_pred, targets)

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            eta_seconds = ((time.time() - start_time) / iteration) * (max_iters - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            if iteration % log_per_iters == 0 and self.save_to_disk:
                logging.info(
                    "Epoch: {:d}/{:d} || Iters: {:d}/{:d} || Lr: {:.6f} || "
                    "Loss: {:.4f} || Cost Time: {} || Estimated Time: {}".format(
                        epoch, epochs, iteration % iters_per_epoch, iters_per_epoch,
                        self.optimizer.param_groups[0]['lr'], losses.item(),
                        str(datetime.timedelta(seconds=int(time.time() - start_time))),
                        eta_string))

            if iteration % self.iters_per_epoch == 0 and self.save_to_disk:
                save_checkpoint(self.args, self.model, epoch, self.optimizer, self.lr_scheduler, is_best=False)

            if not self.args.skip_val and iteration % val_per_iters == 0:
                # self.validation(epoch)
                self.model.eval()
                self.test_2()

                if self.cur_test_2_mIoU > self.best_test_2_mIoU:
                    self.best_test_2_mIoU = self.cur_test_2_mIoU
                    save_checkpoint(self.args, self.model, epoch, self.optimizer, self.lr_scheduler, is_best=True,
                                    best_save_name='best_dp19_model.pth')

                self.model.train()

        total_training_time = time.time() - start_time
        total_training_str = str(datetime.timedelta(seconds=total_training_time))
        logging.info("Total training time: {} ({:.4f}s / it)".format(total_training_str,
                                                                     total_training_time / max_iters))

    def test_2(self):
        self.metric.reset()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        torch.cuda.empty_cache()
        model.eval()
        with torch.no_grad():
            for i, (image, target, filename) in enumerate(self.test_loader_2):
                image = image.to(self.device)
                target = target.to(self.device)
                _, output = model(self.normalize(image))
                self.metric.update(output, target)

            synchronize()
            pixAcc, mIoU, category_iou = self.metric.get(return_category_iou=True)
            logging.info("[TEST DensePASS19 END]  pixAcc: {:.3f}, mIoU: {:.3f}".format(pixAcc * 100, mIoU * 100))
            self.cur_test_2_mIoU = mIoU

        headers = ['class id', 'class name', 'iou']
        table = []
        for i, cls_name in enumerate(self.classes):
            table.append([cls_name, category_iou[i]])
        logging.info('Category iou: \n {}'.format(tabulate(table, headers, tablefmt='grid',
                                                           showindex="always", numalign='center', stralign='center')))
        torch.cuda.empty_cache()
        model.train()


if __name__ == '__main__':
    args = parse_args()
    # get config
    cfg.update_from_file(args.config_file)
    cfg.update_from_list(args.opts)
    cfg.PHASE = 'train' if not args.test else 'test'
    cfg.ROOT_PATH = root_path
    cfg.check_and_freeze()

    # setup python train environment, logger, seed..
    default_setup(args)

    # create a trainer and start train
    trainer = Trainer(args)
    if args.test:
        assert 'pth' in cfg.TEST.TEST_MODEL_PATH, 'please provide test model pth!'
        logging.info('test model......')
    else:
        trainer.train()
