from lib.config import cfg, args
from lib.config.yacs import _to_dict
# from lib.config.parser_util import train_args
from lib.utils.fixseed import fixseed
from lib.networks import make_network
from lib.train import make_trainer, make_optimizer, make_lr_scheduler, make_recorder, set_lr_scheduler
from lib.datasets import make_data_loader
from lib.utils.net_utils import load_model, save_model, load_network, save_trained_config, load_pretrain
from lib.evaluators import make_evaluator
from lib.train.trainers.trainer import Trainer
import torch.multiprocessing
import torch
import torch.distributed as dist
import os
import json
torch.autograd.set_detect_anomaly(True)

if cfg.fix_random:
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def convert_sets_to_lists(obj):
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: convert_sets_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_sets_to_lists(e) for e in obj]
    return obj

def train(cfg):
    # args = train_args()
    # fixseed(args.seed)
    # print(cfg)
    recorder = make_recorder(cfg.save_dir)
    # recorder.report_args(args, name='Args')

    if cfg.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(cfg.save_dir) and not cfg.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(cfg.save_dir))
    elif not os.path.exists(cfg.save_dir):
        os.makedirs(cfg.save_dir)
    args_path = os.path.join(cfg.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(_to_dict(cfg), fw, indent=4, sort_keys=True)

    
    data = make_data_loader(cfg)

    network = make_network(cfg, data)
    # load_network(network, cfg.trained_model_dir, epoch=cfg.test.epoch)
    network.rot2xyz.smpl_model.eval()

    from lib.networks import create_gaussian_diffusion
    diffusion = create_gaussian_diffusion(cfg)

    Trainer(cfg, recorder, network, diffusion, data).run_loop()

    recorder.close()

    # trainer = make_trainer(cfg, network, train_loader)

    # optimizer = make_optimizer(cfg, network)
    # scheduler = make_lr_scheduler(cfg, optimizer)
    # evaluator = make_evaluator(cfg)

    # begin_epoch = load_model(network,
    #                          optimizer,
    #                          scheduler,
    #                          recorder,
    #                          cfg.trained_model_dir,
    #                          resume=cfg.resume)
    # if begin_epoch == 0 and cfg.pretrain != '':
    #     load_pretrain(network, cfg.pretrain)

    # set_lr_scheduler(cfg, scheduler)

    # for epoch in range(begin_epoch, cfg.train.epoch):
    #     recorder.epoch = epoch
    #     if cfg.distributed:
    #         train_loader.batch_sampler.sampler.set_epoch(epoch)

    #     train_loader.dataset.epoch = epoch

    #     trainer.train(epoch, train_loader, optimizer, recorder)
    #     scheduler.step()

    #     if (epoch + 1) % cfg.save_ep == 0 and cfg.local_rank == 0:
    #         save_model(network, optimizer, scheduler, recorder,
    #                    cfg.trained_model_dir, epoch)

    #     if (epoch + 1) % cfg.save_latest_ep == 0 and cfg.local_rank == 0:
    #         save_model(network,
    #                    optimizer,
    #                    scheduler,
    #                    recorder,
    #                    cfg.trained_model_dir,
    #                    epoch,
    #                    last=True)

    #     if (epoch + 1) % cfg.eval_ep == 0 and cfg.local_rank == 0:
    #         trainer.val(epoch, val_loader, evaluator, recorder)

    return network


def test(cfg, network):
    trainer = make_trainer(cfg, network)
    val_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)
    epoch = load_network(network,
                         cfg.trained_model_dir,
                         resume=cfg.resume,
                         epoch=cfg.test.epoch)
    trainer.val(epoch, val_loader, evaluator)

def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

def main():
    train(cfg)
#     if cfg.distributed:
#         cfg.local_rank = int(os.environ['RANK']) % torch.cuda.device_count()
#         torch.cuda.set_device(cfg.local_rank)
#         torch.distributed.init_process_group(backend="nccl",
#                                              init_method="env://")
#         synchronize()

#     if args.test:
#         test(cfg)
#     else:
#         train(cfg)
#     if cfg.local_rank == 0:
#         print('Success!')
#         print('='*80)
#     os.system('kill -9 {}'.format(os.getpid()))


if __name__ == "__main__":
    main()
