import tensorflow as tf

from spektral.datasets import Citation
from spektral.transforms import LayerPreprocess, AdjToSpTensor
from spektral.layers import GCNConv
from spektral.data import SingleLoader

import numpy as np


def prepare_dataset(dataset):
    # LayerPreprocesses adds self loops and normalizes the adjacency matrix
    dataset = Citation(
        dataset,
        normalize_x=True,
        transforms=[LayerPreprocess(GCNConv), AdjToSpTensor()],
    )

    print("Dataset: ", dataset.name)
    print("Size of train set:", dataset.mask_tr.sum().item())
    print("Size of val set:", dataset.mask_va.sum().item())
    print("Size of test set:", dataset.mask_te.sum().item())
    print("Num classes:", dataset.n_labels)
    print("Num features:", dataset.n_node_features)

    return dataset


def mask_to_weights(mask):
    return mask.astype(np.float32) / np.count_nonzero(mask)


def generate_data_loaders(dataset):
    # Extract train, test, validation sets using masks
    weights_tr, weights_va, weights_te = (
        mask_to_weights(mask)
        for mask in (dataset.mask_tr, dataset.mask_va, dataset.mask_te)
    )
    loader_tr = SingleLoader(dataset, sample_weights=weights_tr)
    loader_va = SingleLoader(dataset, sample_weights=weights_va)
    loader_te = SingleLoader(dataset, sample_weights=weights_te)

    return loader_tr, loader_va, loader_te
