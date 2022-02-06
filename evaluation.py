import argparse
import os
import time
import warnings
warnings.filterwarnings("ignore")

import yaml
import pandas as pd
import numpy as np

from feature_eval import calculate_nsc_and_nscb
from generate_features import get_embeddings, get_data


# create moa meta data file
def get_meta(sc_meta_path):
    meta_sc = pd.read_csv(sc_meta_path)
    meta_sc['Metadata_Plate'] = [s.split("-")[0] for s in meta_sc['Image_Name']]
    meta_sc['Metadata_Well'] = [s.split("-")[1] for s in meta_sc['Image_Name']]
    meta_sc['Metadata_Site'] = [s.split("-")[2][:2] for s in meta_sc['Image_Name']]
    meta_sc['compound'] = [s.split("_")[0] for s in meta_sc['Class_Name']]
    meta_sc['concentration'] = [float(s.split("_")[1]) for s in meta_sc['Class_Name']]
    meta_sc['Replicate'] = 1

    moa = pd.read_csv("https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_moa.csv")
    moa['concentration'] = moa['concentration'].astype(str)
    moa['Class_Name'] = moa[['compound', 'concentration']].agg('_'.join, axis=1)
    moa_dict = dict(zip(moa['Class_Name'], moa['moa']))
    meta_sc['moa'] = [moa_dict[c] for c in meta_sc['Class_Name']]

    return meta_sc


def all_checkpoints_exist(checkpoint_list):
    for checkpoint in checkpoint_list:
        if not os.path.isfile(checkpoint):
            return False
    return True


def checkpoint_exists(checkpoint):
    if not os.path.isfile(checkpoint):
        return False
    return True


def nsc_nscb(config_file, epoch_list):
    config = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)

    # Pre check for checkpoints
    checkpoints_dir = os.path.join(config['run_dir'], "checkpoints")
    status = all_checkpoints_exist([os.path.join(checkpoints_dir, f"ckpt_epoch_{e}.pth") for e in epoch_list])
    print(f"Checking directory {checkpoints_dir} for epochs {epoch_list} to all exist: {status}")

    loader = get_data(config)
    meta = get_meta(config['eval_dataset']['path'])

    print("Evaluating", config['run_dir'])
    for epoch in epoch_list:
        model_path = os.path.join(checkpoints_dir, f"ckpt_epoch_{epoch}.pth")

        print(f"Waiting for the epoch {epoch} checkpoint to become available in {checkpoints_dir} ...")
        while not checkpoint_exists(model_path):
            time.sleep(60)

        features = get_embeddings(config, model_path, loader)
        os.makedirs(os.path.join(config['run_dir'], "features"), exist_ok=True)
        np.save(os.path.join(config['run_dir'], f"features/features_{epoch}.npy"), features)

        plots_dir = os.path.join(config['run_dir'], "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # without TVN transformation
        plot_file_structure = os.path.join(plots_dir, f"{{}}_epoch_{epoch}_untransformed.jpg")
        nsc, nscb = calculate_nsc_and_nscb(features=features,
                                           meta=meta,
                                           plot_file_structure=plot_file_structure,
                                           DO_WHITENING=False,
                                           DO_CORAL=False)

        # with whitening transformation
        plot_file_structure = os.path.join(plots_dir, f"{{}}_epoch_{epoch}_whitened.jpg")
        w_nsc, w_nscb = calculate_nsc_and_nscb(features=features,
                                               meta=meta,
                                               plot_file_structure=plot_file_structure,
                                               DO_WHITENING=True,
                                               DO_CORAL=False)

        # with TVN transformation
        plot_file_structure = os.path.join(plots_dir, f"{{}}_epoch_{epoch}_TVN.jpg")
        tvn_nsc, tvn_nscb = calculate_nsc_and_nscb(features=features,
                                                   meta=meta,
                                                   plot_file_structure=plot_file_structure,
                                                   DO_WHITENING=True,
                                                   DO_CORAL=True)

        print(f"Results for {config['run_dir']} epoch {epoch};")
        print(f"NSC:{nsc} NSCB:{nscb}")
        print(f"whitening-NSC:{w_nsc} whitening-NSCB:{w_nscb}")
        print(f"TVN-NSC:{tvn_nsc} TVN-NSCB:{tvn_nscb}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cell Profile Evaluation")
    parser.add_argument("config_file", type=str, help="YAML config file")
    parser.add_argument("--epochs", type=str, help="list of epochs to load checkpoint")
    args = parser.parse_args()
    epoch_list = [int(e) for e in args.epochs.split(",")]
    nsc_nscb(args.config_file, epoch_list)
