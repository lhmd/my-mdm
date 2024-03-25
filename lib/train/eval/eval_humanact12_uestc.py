"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import os
import torch
import re

from lib.utils import dist_util
from lib.networks.mdm.cfg_sampler import ClassifierFreeSampleModel
from lib.datasets.make_dataset import get_data_loader
from lib.train.eval.a2m.tools import save_metrics
from lib.config import cfg
# from lib.config.parser_util import evaluation_parser
from lib.utils.fixseed import fixseed
from lib.networks.make_network import make_network, create_gaussian_diffusion, load_model_wo_clip


def evaluate(args, model, diffusion, data):
    scale = None
    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)  # wrapping model with the classifier-free sampler
        scale = {
            'action': torch.ones(args.batch_size) * args.guidance_param,
        }
    model.to(dist_util.dev())
    model.eval()  # disable random masking


    folder, ckpt_name = os.path.split(args.model_path)
    if args.dataset == "humanact12":
        from lib.train.eval.a2m.gru_eval import evaluate
        eval_results = evaluate(args, model, diffusion, data)
    elif args.dataset == "uestc":
        from lib.train.eval.a2m.stgcn_eval import evaluate
        eval_results = evaluate(args, model, diffusion, data)
    else:
        raise NotImplementedError("This dataset is not supported.")

    # save results
    iter = int(re.findall('\d+', ckpt_name)[0])
    scale = 1 if scale is None else scale['action'][0].item()
    scale = str(scale).replace('.', 'p')
    metricname = "evaluation_results_iter{}_samp{}_scale{}_a2m.yaml".format(iter, args.num_samples, scale)
    evalpath = os.path.join(folder, metricname)
    print(f"Saving evaluation: {evalpath}")
    save_metrics(evalpath, eval_results)

    return eval_results


def main(cfg):
    # args = evaluation_parser()
    # fixseed(args.seed)
    args = cfg
    dist_util.setup_dist(args.device)

    print(f'Eval mode [{args.eval_mode}]')
    assert args.eval_mode in ['debug', 'full'], f'eval_mode {args.eval_mode} is not supported for dataset {args.dataset}'
    if args.eval_mode == 'debug':
        args.num_samples = 10
        args.num_seeds = 2
    else:
        args.num_samples = 1000
        args.num_seeds = 20

    data_loader = get_data_loader(name=args.dataset, num_frames=60, batch_size=args.batch_size,)

    print("creating model and diffusion...")
    # model, diffusion = create_model_and_diffusion(args, data_loader)
    model = make_network(args, data_loader)
    diffusion = create_gaussian_diffusion(args)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    eval_results = evaluate(args, model, diffusion, data_loader.dataset)

    fid_to_print = {k : sum([float(vv) for vv in v])/len(v) for k, v in eval_results['feats'].items() if 'fid' in k and 'gen' in k}
    print(fid_to_print)

if __name__ == '__main__':
    main()
