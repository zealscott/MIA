import os
import pickle
import yaml

import numpy as np
import torch.nn as nn
from models.densenet import create_densenet121
from models.mobilenetv2 import create_mobilenetv2
from models.resnet import create_wideresnet
from models.vgg import create_vgg16
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST
from pytorch_cinic.dataset import CINIC10
from torch.utils.data import Subset


def get_balanced_cinic10_indices(dataset, num_classes, samples_per_class, seed=42):
    """
    Helper function to get balanced subset indices for CINIC10.
    Returns sorted indices for consistent sampling.
    """
    np.random.seed(seed)

    # Get all labels from the dataset
    all_labels = [dataset[i][1] for i in range(len(dataset))]

    # Group indices by class
    class_indices = [[] for _ in range(num_classes)]
    for idx, label in enumerate(all_labels):
        class_indices[label].append(idx)

    # Sample equal number from each class
    selected_indices = []
    for class_idx_list in class_indices:
        indices = np.random.choice(class_idx_list, samples_per_class, replace=False)
        selected_indices.extend(indices)

    return sorted(selected_indices)


def load_dataset(args):
    """
    load augmented training/test datasets
    """
    tv_dataset = get_dataset(args)

    # Decide transforms:
    # MNIST / fMNIST are grayscale 28x28; use the transforms for MNIST
    # CIFAR10 / CIFAR100 / CINIC10 are 32x32 color; use the CIFAR-like transforms
    if args.dataset in ["mnist", "fmnist"]:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(args.data_mean, args.data_std),
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(args.data_mean, args.data_std),
            ]
        )
    else:
        # CIFAR and CINIC are 32x32 color
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(args.data_mean, args.data_std),
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(args.data_mean, args.data_std),
            ]
        )

    # CINIC-10 logic to load "train" and "test" sets.
    # If it's not CINIC-10, we just pass train=True/False as usual.
    if args.dataset == "cinic10":
        train_ds = tv_dataset(
            root=os.path.join(args.data_dir, "cinic10"), partition="train", download=True, transform=train_transform
        )
        test_ds = tv_dataset(
            root=os.path.join(args.data_dir, "cinic10"), partition="test", download=True, transform=test_transform
        )
        # Create subset datasets
        train_indices = get_balanced_cinic10_indices(train_ds, args.num_classes, samples_per_class=5000)
        test_indices = get_balanced_cinic10_indices(test_ds, args.num_classes, samples_per_class=1000)
        train_ds = Subset(train_ds, train_indices)
        test_ds = Subset(test_ds, test_indices)
    else:
        train_ds = tv_dataset(root=args.data_dir, train=True, download=True, transform=train_transform)
        test_ds = tv_dataset(root=args.data_dir, train=False, download=True, transform=test_transform)

    return train_ds, test_ds


def load_labels(args):
    """
    get the label w.r.t dataset
    """
    if args.dataset == "cifar10":
        train_ds = CIFAR10(root=args.data_dir, train=True, download=True)
    elif args.dataset == "cifar100":
        train_ds = CIFAR100(root=args.data_dir, train=True, download=True)
    elif args.dataset == "mnist":
        train_ds = MNIST(root=args.data_dir, train=True, download=True)
    elif args.dataset == "fmnist":
        train_ds = FashionMNIST(root=args.data_dir, train=True, download=True)
    elif args.dataset == "cinic10":
        # The CINIC-10 package handles splitting inside the class constructor
        train_ds = CINIC10(root=os.path.join(args.data_dir, "cinic10"), partition="train", download=True)
        train_indices = get_balanced_cinic10_indices(train_ds, args.num_classes, samples_per_class=5000)
        return np.array([train_ds[i][1] for i in train_indices])
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")
    return np.array(train_ds.targets)


def get_dataset(args):
    """
    Returns the appropriate dataset class (not an instance) and sets
    dataset-specific mean, std, and num_classes in 'args'.
    """
    if args.dataset == "cifar10":
        args.data_mean = (0.4914, 0.4822, 0.4465)
        args.data_std = (0.2023, 0.1994, 0.2010)
        args.num_classes = 10

        return CIFAR10
    elif args.dataset == "cifar100":
        args.data_mean = (0.5071, 0.4867, 0.4408)
        args.data_std = (0.2675, 0.2565, 0.2761)
        args.num_classes = 100

        return CIFAR100
    elif args.dataset == "mnist":
        args.data_mean = (0.1307,)
        args.data_std = (0.3081,)
        args.num_classes = 10

        return MNIST
    elif args.dataset == "fmnist":
        # Example mean/std commonly used for FashionMNIST
        args.data_mean = (0.2860,)
        args.data_std = (0.3530,)
        args.num_classes = 10

        return FashionMNIST
    elif args.dataset == "cinic10":
        args.data_mean = (0.4789, 0.4723, 0.4305)
        args.data_std = (0.2421, 0.2383, 0.2587)
        args.num_classes = 10

        return CINIC10
    else:
        raise NotImplementedError(f"Dataset '{args.dataset}' not implemented.")


def load_model(args):
    """
    Load model with appropriate configurations based on dataset and model type.
    """
    if args.model_type == "resnet":
        m = create_wideresnet(dataset=args.dataset, num_classes=args.num_classes)
    elif args.model_type == "densenet":
        m = create_densenet121(dataset=args.dataset, num_classes=args.num_classes)
    elif args.model_type == "mobilenet":
        m = create_mobilenetv2(dataset=args.dataset, num_classes=args.num_classes)
    elif args.model_type == "vgg":
        m = create_vgg16(dataset=args.dataset, num_classes=args.num_classes)

    return m


def load_mask(keep_path):
    print(f"loading mask from {keep_path} ...")
    if "r0" in keep_path:
        print("round 1 does not have keep mask, skip ...")
        return [], []
    if not os.path.exists(keep_path):
        raise ValueError(f"keep_path {keep_path} does not exist")
    else:
        anchored = pickle.load(open(keep_path, "rb"))
        force_in = [id for id, value in anchored.items() if value == 1]
        force_out = [id for id, value in anchored.items() if value == 0]

        print(
            f"size of anchored: {len(anchored)}, "
            f"size of force_in: {len(force_in)}, size of force_out: {len(force_out)}"
        )

        return force_in, force_out


def load_config(args):
    print(f"loading config for {args.dataset} ...")
    with open(f"config/{args.dataset}.yml", "r") as file:
        config = yaml.safe_load(file)
    for key, value in config.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)

    args.savedir = f"{args.dataset}/r{args.cur_round}_exp/"
    # the keep mask for the previous round
    args.keep_path = f"{args.dataset}/anchor_mask_r{args.cur_round-1}.pkl"
    os.makedirs(args.savedir, exist_ok=True)
