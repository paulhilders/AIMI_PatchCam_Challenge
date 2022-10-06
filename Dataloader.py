import torch
from torch.utils.data import Dataset
import h5py
import matplotlib.pyplot as plt

class PCAMDataset(Dataset):
    def __init__(self, h5pyfile_x, h5pyfile_y, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.pcam_x_frame = h5py.File(h5pyfile_x, 'r')['x']
        print(self)
        self.pcam_y_frame = h5py.File(h5pyfile_y, 'r')['y']
        self.transform = transform

    def __len__(self):
        return len(self.pcam_x_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.pcam_x_frame[idx]
        label = self.pcam_y_frame[idx]

        image = torch.from_numpy(image).long()
        label = torch.from_numpy(label).long()

        image = image.permute(2, 0, 1).byte() / 255
        label = label.squeeze()

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'label': label}

        return sample


def getDataLoaders(transform=None):
    train_dataset = PCAMDataset('pcamv1/camelyonpatch_level_2_split_train_x.h5', 'pcamv1/camelyonpatch_level_2_split_train_y.h5', transform=transform)
    valid_dataset = PCAMDataset('pcamv1/camelyonpatch_level_2_split_valid_x.h5', 'pcamv1/camelyonpatch_level_2_split_valid_y.h5', transform=transform)
    test_dataset = PCAMDataset('pcamv1/camelyonpatch_level_2_split_test_x.h5', 'pcamv1/camelyonpatch_level_2_split_test_y.h5', transform=transform)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   shuffle=True, batch_size=64)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                   shuffle=True, batch_size=64)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset,
                                                   shuffle=True, batch_size=64)
    return train_dataloader, test_dataloader, valid_dataloader


def test_print(train_dataloader):
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")


def test_print_with_label(train_dataloader, label=1):
    fig, axs = plt.subplots(3, 3)
    fig.suptitle(f'Example data with label {label}')
    for x in range(3):
        for y in range(3):
            label_true = False
            while not label_true:
                train_features, train_label = next(iter(train_dataloader))
                if train_label[0] == label:
                    label_true = True
            img = train_features[0].squeeze()
            axs[x,y].imshow(img, cmap="gray")
            axs[x,y].axis('off')
    plt.savefig(f'./imgs/exampledata_label{label}', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    train_dataloader, test_dataloader, valid_dataloader = getDataLoaders(transform=None)
    # test_print(train_dataloader)
    test_print_with_label(train_dataloader, label=1)
    test_print_with_label(train_dataloader, label=0)
