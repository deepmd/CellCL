import argparse
import os

import torch
from torch.utils.data import DataLoader
import yaml
import numpy as np
from tqdm import tqdm

from data import CellDataset, get_eval_transform
from models import get_model


def get_data(config):
    data_transforms = get_eval_transform()
    dataset = CellDataset(config['eval_dataset']['path'],
                          config['eval_dataset']['root_dir'],
                          eval(config['dataset']['input_shape']),
                          config['dataset']['preload'],
                          transform=data_transforms)
    loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        num_workers=1,
        shuffle=False)
    return loader


@torch.no_grad()
def convert_tensor_to_np(model, data_loader):
    train_feature_vector = []
    train_labels_vector = []
    model.eval()
    for batch_x, batch_y in tqdm(data_loader):
        batch_x = batch_x.cuda()
        train_labels_vector.extend(batch_y)
        features = model(batch_x)
        train_feature_vector.extend(features.cpu().detach().numpy())

    train_feature_vector = np.array(train_feature_vector)
    train_labels_vector = np.array(train_labels_vector)

    return train_feature_vector, train_labels_vector


def _create_model(model_name, model_path):
    model = get_model(model_name)[0].cuda()
    checkpoint = torch.load(model_path)
    encoder_state_dict = dict()
    for k, v in checkpoint['state_dict'].items():
        if k.startswith('encoder_q.'):
            encoder_state_dict[k[10:]] = v
    model.load_state_dict(encoder_state_dict)
    return model


def main(config_file, epoch):
    config = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)
    model_path = os.path.join(config['run_dir'], f"checkpoints/ckpt_epoch_{epoch}.pth")
    model = _create_model(config['isd']['base_model'], model_path)

    loader = get_data(config)
    X, Y = convert_tensor_to_np(model, loader)

    os.makedirs(os.path.join(config['run_dir'], "features"), exist_ok=True)
    np.save(os.path.join(config['run_dir'], f"features/features_{epoch}.npy"), X)


def get_embeddings(config, model_path, loader):
    model = _create_model(config['isd']['base_model'], model_path)
    X, Y = convert_tensor_to_np(model, loader)
    return X


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cell Profile Generating Features")
    parser.add_argument("config_file", type=str, help="YAML config file")
    parser.add_argument("--epoch", type=int, help="epoch to load checkpoint")
    args = parser.parse_args()
    main(args.config_file, args.epoch)
