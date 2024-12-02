import torch
from torch.utils.data import Dataset, DataLoader, random_split
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
    #train_ds = AckermanDataset(os.path.join(data_path, 'AckermanDataset10K_train.csv'))
    #test_ds = AckermanDataset(os.path.join(data_path, 'AckermanDataset10K_test.csv'))
    train_ds = AckermanDataset(os.path.join(data_path, 'overfitting_train.csv'))
    test_ds = AckermanDataset(os.path.join(data_path, 'overfitting_test.csv'))
    train_dl = DataLoader(train_ds, **dl_args)
    test_dl = DataLoader(test_ds, **dl_args)

    return train_ds, train_dl, test_ds, test_dl


class AckermanDataset(Dataset):
    def __init__(self, data_path):
        data = torch.from_numpy(pandas.read_csv(data_path).values)
        X = data[:, 0:3]
        y = data[:, 3:]

        self.X = X
        self.y = y
        # self.X = StandardScaler().fit(X)
        # self.y = StandardScaler().fit(y)
        # self.X = torch.Tensor(self.X.transform(X)).to(torch.float64)
        # self.y = torch.Tensor(self.y.transform(y)).to(torch.float64)

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, item):
        """
        :param item: index of requested item
        :return: the item and the label
        """
        return self.X[item], self.y[item]