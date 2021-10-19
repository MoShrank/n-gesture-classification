import tonic
import tonic.transforms as transforms
import torch


def get_dvs_datasets(
    time_factor: float = 0.9,
    spatial_factor: float = 0.75,
    data_path: str = "../data/",
    download: bool = False,
):
    transform = tonic.transforms.Compose(
        [
            transforms.Downsample(
                time_factor=time_factor, spatial_factor=spatial_factor
            ),
            transforms.ToSparseTensor(merge_polarities=True),
        ]
    )

    trainset = tonic.datasets.DVSGesture(
        save_to=data_path,
        download=download,
        transform=transform,
        train=True,
    )
    testset = tonic.datasets.DVSGesture(
        save_to=data_path,
        download=download,
        transform=transform,
        train=False,
    )

    return trainset, testset


def get_dvs_data_generator(
    batch_size: int = 16,
    time_factor: float = 0.9,
    spatial_factor: float = 0.75,
    data_path: float = "../data/",
    download: bool = False,
):

    trainset, testset = get_dvs_datasets(
        time_factor, spatial_factor, data_path, download
    )

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        collate_fn=tonic.utils.pad_tensors,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        collate_fn=tonic.utils.pad_tensors,
        shuffle=False,
    )

    return train_loader, test_loader
