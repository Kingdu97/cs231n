import torch

from torch.utils.data import Dataset, DataLoader


class MnistDataset(Dataset):        # custom dataset을 만들것

    def __init__(self, data, labels, flatten=True):# 28*28을 벡터로 만들지, 그대로 이미지로 갈지
        self.data = data
        self.labels = labels
        self.flatten = flatten

        super().__init__()

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        if self.flatten:        # flatten 됐기에 (28*28) -> (784,)
            x = x.view(-1)

        return x, y


def load_mnist(is_train=True, flatten=True):
    from torchvision import datasets, transforms

    dataset = datasets.MNIST(
        '../data', train=is_train, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
    )

    x = dataset.data.float() / 255.
    y = dataset.targets

    if flatten:
        x = x.view(x.size(0), -1)

    return x, y


def get_loaders(config):
    x, y = load_mnist(is_train=True, flatten=False)     #ㅣxㅣ= (60000,28,28)

    train_cnt = int(x.size(0) * config.train_ratio)
    valid_cnt = x.size(0) - train_cnt

    flatten = True if config.model == 'fc' else False

    # Shuffle dataset to split into train/valid set.
    indices = torch.randperm(x.size(0))
    train_x, valid_x = torch.index_select(
        x,
        dim=0,
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)       # train_x = (48000,28,28) valid_x = (12000,28,28)
    train_y, valid_y = torch.index_select(
        y,
        dim=0,
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)

    train_loader = DataLoader(
        dataset=MnistDataset(train_x, train_y, flatten=flatten),
        batch_size=config.batch_size,
        shuffle=True,                   # training은 무조건 shuffle = True
    )
    valid_loader = DataLoader(
        dataset=MnistDataset(valid_x, valid_y, flatten=flatten),
        batch_size=config.batch_size,
        shuffle=True,
    )

    test_x, test_y = load_mnist(is_train=False, flatten=False)
    test_loader = DataLoader(
        dataset=MnistDataset(test_x, test_y, flatten=flatten),
        batch_size=config.batch_size,
        shuffle=False,              # test set은 기본적으로 shuffle = False
    )

    return train_loader, valid_loader, test_loader
