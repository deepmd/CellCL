import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from torch import DoubleTensor

from .cell_dataset import CellDataset


class DataSetWrapper(object):

    def __init__(self, batch_size, path, root_dir, num_workers, input_shape, preload, transform,
                 valid_size=0, sampler="random", **kwargs):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.sampler = sampler
        input_shape = eval(input_shape)
        self.dataset = CellDataset(path, root_dir, input_shape, preload, transform)

    def get_train_loader(self):
        train_loader = DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                  drop_last=True, shuffle=True)
        return train_loader

    def get_train_valid_loaders(self):
        # obtain training indices that will be used for validation
        num_train = len(self.dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        if self.sampler == "random":
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)
        elif self.sampler == "weighted":
            labels_df = self.dataset.labels.to_frame()
            weights = 1. / (int(labels_df.nunique()) * labels_df.groupby('Target')['Target'].transform('count'))
            # target 32 is "DMSO_0.0"
            epoch_size = len(labels_df[labels_df['Target'] != 32]) * int(labels_df.nunique()) / (int(labels_df.nunique())-1)

            train_sampler = WeightedRandomSampler(weights=DoubleTensor(list(weights[train_idx])),
                                                  num_samples=int(np.floor((1-self.valid_size) * epoch_size)),
                                                  replacement=False)
            valid_sampler = WeightedRandomSampler(weights=DoubleTensor(list(weights[valid_idx])),
                                                  num_samples=int(np.floor(self.valid_size * epoch_size)),
                                                  replacement=False)
        else:
            raise Exception(f"Sampler {self.sampler} is not supported.")

        train_loader = DataLoader(self.dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True)
        valid_loader = DataLoader(self.dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True)
        return train_loader, valid_loader
