import builtins
import os
import sys
import time
import argparse

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from data import DataSetWrapper, get_train_transform, get_eval_transform
from feature_eval import RFClassifier
from isd import ISD, KLD
from models import get_model
from utils import get_logger, AverageMeter


def main(config_file):
    config = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)

    os.makedirs(os.path.join(config['run_dir'], "checkpoints"), exist_ok=True)

    logger = get_logger(logpath=os.path.join(config['run_dir'], "training.log"))
    def print_pass(*args):
        logger.info(*args)
    builtins.print = print_pass
    writer = SummaryWriter(log_dir=os.path.join(config['run_dir'], "tensorboard"))

    num_gpus = torch.cuda.device_count()

    print("Config: {}\n{}\n{}{}".format(config_file, "-"*40, yaml.dump(config, sort_keys=False), "-"*40))
    print("Number of gpus: {}".format(num_gpus))

    # set deterministic training for reproducibility
    if 'seed' in config:
        import random
        import numpy
        random.seed(config['seed'])
        numpy.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        # setting the following flags degrade performance considerably!
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    dataset = DataSetWrapper(batch_size=num_gpus * config['batch_size'],
                             transform=get_train_transform(config['augmentation']), **config['dataset'])
    train_loader = dataset.get_train_loader()
    if 'eval_freq' in config:
        eval_dataset = DataSetWrapper(batch_size=num_gpus * config['batch_size'],
                                      transform=get_eval_transform(), **config['eval_dataset'])

    isd = ISD(base_encoder=lambda: get_model(config['isd']['base_model']), **config['isd'])
    isd = torch.nn.DataParallel(isd).cuda()

    criterion = KLD().cuda()

    params = [p for p in isd.parameters() if p.requires_grad]
    if 'sgd' in config['optim']:
        optimizer = torch.optim.SGD(params, **config['optim']['sgd'])
    else:
        raise ValueError("Unknown optimizer was specified in config.")
    if config['optim']['scheduler'] == "cos":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs']*len(train_loader))
    else:
        raise ValueError("Unknown scheduler was specified in config.")

    start_epoch = 1

    if 'resume' in config:
        print("==> resume from checkpoint: {}".format(config['resume']))
        ckpt = torch.load(config['resume'])
        print("==> resume from epoch: {}".format(ckpt['epoch']))
        isd.module.load_state_dict(ckpt['state_dict'], strict=True)
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1

    # routine
    for epoch in range(start_epoch, config['epochs'] + 1):

        print("==> training...")
        time1 = time.time()
        train_student(epoch, train_loader, isd, criterion, optimizer, lr_scheduler, config['print_freq'], writer)
        time2 = time.time()
        print("epoch {}, total time {:.2f}".format(epoch, time2 - time1))

        # train linear model if requested
        if 'eval_freq' in config and epoch % config['eval_freq'] == 0:
            print("==> evaluating...")
            time1 = time.time()
            score_eval = eval_classifier(isd, eval_dataset)
            time2 = time.time()
            print("epoch {}, classifier accuracy {:.3f}, total time {:.2f}".format(epoch, score_eval, time2 - time1))
            writer.add_scalar("classifier_accuracy", score_eval, epoch)

        # saving the model
        if epoch % config['save_freq'] == 0:
            print("==> saving...")
            state = {
                'opt': config,
                'state_dict': isd.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict(),
                'epoch': epoch
            }
            save_file = os.path.join(config['run_dir'], "checkpoints/ckpt_epoch_{epoch}.pth".format(epoch=epoch))
            torch.save(state, save_file)
            # help release GPU memory
            del state
            torch.cuda.empty_cache()


def train_student(epoch, train_loader, isd, criterion, optimizer, lr_scheduler, print_freq, writer):
    """
    one epoch training for CompReSS
    """
    isd.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()

    end = time.time()
    for idx, ((im_q, im_k), _) in enumerate(train_loader):
        data_time.update(time.time() - end)
        im_q = im_q.cuda(non_blocking=True)
        im_k = im_k.cuda(non_blocking=True)

        # ===================forward=====================
        sim_q, sim_k = isd(im_q=im_q, im_k=im_k)
        loss = criterion(inputs=sim_q, targets=sim_k)

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr = optimizer.param_groups[0]['lr']
        lr_scheduler.step()

        # ===================meters=====================
        loss_meter.update(loss.item(), im_q.size(0))

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % print_freq == 0:
            print("Train: [{0}][{1}/{2}]\t"
                  "BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                  "DT {data_time.val:.3f} ({data_time.avg:.3f})\t"
                  "LR {lr:.4f}\t"
                  "Loss {loss.val:.3f} ({loss.avg:.3f})\t".format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=loss_meter, lr=lr))
            sys.stdout.flush()
            global_step = (epoch-1)*len(train_loader) + idx + 1
            writer.add_scalar("loss", loss_meter.avg, global_step)
            writer.add_scalar("learning_rate", lr, global_step)

    return loss_meter.avg


@torch.no_grad()
def eval_classifier(isd, eval_dataset):
    isd.eval()
    classifier = RFClassifier(isd)
    train_loader, validate_loader = eval_dataset.get_train_valid_loaders()
    print("RF classifier training started.")
    classifier.train(train_loader)
    print("Training classifier done.")
    score_eval = classifier.test(validate_loader)
    return score_eval


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description="ISD Cell Profile Training")
    parser.add_argument("config_file", type=str, help="YAML config file")
    args = parser.parse_args()
    main(args.config_file)
