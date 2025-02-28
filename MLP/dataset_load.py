import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas
import os


def create_dataloaders(data_path: str = 'data', ratio=0.2, **dl_args):
    """
    :param data_path: path to the location of the dataset
    :param ratio: the ratio to split to test
    :param dl_args: arguments that will be passed to the dataloader (for example: batch_size=32 to change the batch size)
    :return: DataLoaders for the train and test sets
    """
    train_ds = AckermanDataset(os.path.join(f'{data_path}_train.csv'))
    test_ds = AckermanDataset(os.path.join(f'{data_path}_test.csv'))
    
    x_min = torch.tensor(np.expand_dims(np.min(np.array([np.min(train_ds.X.numpy(), axis=0), np.min(test_ds.X.numpy(), axis=0)]), axis=0), axis=0))
    x_max = torch.tensor(np.expand_dims(np.max(np.array([np.max(train_ds.X.numpy(), axis=0), np.max(test_ds.X.numpy(), axis=0)]), axis=0), axis=0))
    
    train_dl = DataLoader(train_ds, **dl_args)
    test_dl = DataLoader(test_ds, **dl_args)

    return train_ds, train_dl, test_ds, test_dl, x_min, x_max


class AckermanDataset(Dataset):
    def __init__(self, data_path):
        data = torch.from_numpy(pandas.read_csv(data_path).values)
        X = data[:, 0:3]
        y = data[:, 3:]

        self.X = X
        self.y = y
        

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, item):
        """
        :param item: index of requested item
        :return: the item and the label
        """
        return self.X[item], self.y[item]