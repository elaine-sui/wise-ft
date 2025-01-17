import os

import sys

sys.path.append(os.getcwd())

import numpy as np

import torch

from src.models.eval import evaluate
from src.models.finetune import finetune
from src.models.modeling import ClassificationHead, ImageEncoder, ImageClassifier
from src.models.utils import fisher_load
from src.models.zeroshot import get_zeroshot_classifier
from src.args import parse_arguments

from datetime import datetime
import wandb

def modify_args(args):
    date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    args.datetime = date_str

    model_name_safe = args.model.replace('/', '-')

    train_dataset_str = args.train_dataset

    args.exp_name = model_name_safe
    if args.params_to_unfreeze is not None:
        unfreeze_str = '_'.join(args.params_to_unfreeze)
        args.exp_name += f"_unfreeze_{unfreeze_str}"
    
    if args.restrict_grad_dims:
        args.exp_name += f"_restrict_k_{args.k}_dims"

    args.exp_name += f"_{train_dataset_str}/{date_str}"

    args.save_dir = os.path.join(args.save, train_dataset_str, args.datetime)
    args.save_zero_shot_dir = os.path.join(args.save, train_dataset_str)
    args.results_db = os.path.join(args.save_dir, args.results_db)

    return args

def wandb_init(args):
    wandb.init(
        name=args.exp_name,
        project="wise-ft",
        config=args
    )


def _merge(alpha, theta_0, theta_1, fishers, fisher_floor):
    if fishers is None:
        # interpolate between all weights in the checkpoints
        return {
            key: (1 - alpha) * theta_0[key] + alpha * theta_1[key]
            for key in theta_0.keys()
        }

    fisher_0, fisher_1 = fishers

    theta = {}
    for key in theta_0.keys():
        # Make sure that either we have a Fisher for this variable for
        # both checkpoints or none of the checkpoints. Default to regular
        # interpolation if no Fisher is found.
        assert (key in fisher_0) == (key in fisher_1)
        ones = torch.ones_like(theta_0[key])
        f_0 = torch.maximum(fisher_0.get(key, ones), fisher_floor * ones)
        f_1 = torch.maximum(fisher_1.get(key, ones), fisher_floor * ones)

        c_0 = (1 - alpha) * f_0
        c_1 = alpha * f_1

        theta[key] = (c_0 * theta_0[key] + c_1 * theta_1[key]) / (c_0 + c_1)

    return theta


def wise_ft(args):
    assert args.save is not None, 'Please provide a path to store models'

    # Modify args
    args = modify_args(args)

    # Wandb init
    if args.wandb:
        wandb_init(args)
    
    if args.load is None:
        # Build and save zero-shot model
        image_encoder = ImageEncoder(args, keep_lang=True)
        zeroshot_checkpoint = os.path.join(args.save_zero_shot_dir, 'zeroshot.pt')

        if not os.path.exists(zeroshot_checkpoint):
            classification_head = get_zeroshot_classifier(args, image_encoder.model)
            delattr(image_encoder.model, 'transformer')
            classifier = ImageClassifier(image_encoder, classification_head, process_images=False)
            classifier.save(zeroshot_checkpoint)

        # Standard fine-tuning
        args.load = zeroshot_checkpoint
        finetuned_checkpoint = finetune(args)
    else:
        # No need to compute things from stratch
        assert len(args.load) == 2
        zeroshot_checkpoint, finetuned_checkpoint = args.load

    # Load models
    zeroshot = ImageClassifier.load(zeroshot_checkpoint)
    finetuned = ImageClassifier.load(finetuned_checkpoint)
    theta_0 = {k: v.clone() for k, v in zeroshot.state_dict().items()}
    theta_1 = {k: v.clone() for k, v in finetuned.state_dict().items()}
    del zeroshot

    if args.fisher is None:
        fishers = None
    else:
        fisher_0_file, fisher_1_file = args.fisher
        fisher_0 = fisher_load(os.path.expanduser(fisher_0_file))
        fisher_1 = fisher_load(os.path.expanduser(fisher_1_file))
        fishers = fisher_0, fisher_1

    # make sure checkpoints are compatible
    assert set(theta_0.keys()) == set(theta_1.keys())

    alphas = args.alpha
    for alpha in alphas:
        args.alpha = alpha

        theta = _merge(alpha, theta_0, theta_1, fishers, args.fisher_floor)

        # update the model (in-place) acccording to the new weights
        finetuned.load_state_dict(theta)

        # save model
        finetuned.save(os.path.join(args.save_dir, f'wise_ft_alpha={alpha:.3f}.pt'))

        # evaluate
        args.wandb = False # Don't log the actual metrics (just log console)
        evaluate(finetuned, args)


if __name__ == '__main__':
    args = parse_arguments()
    wise_ft(args)
