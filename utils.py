import tensorflow as tf

from spektral.datasets import Citation
from spektral.transforms import LayerPreprocess, AdjToSpTensor
from spektral.layers import GCNConv
from spektral.data import SingleLoader

import numpy as np
import pandas as pd
from tqdm import trange
import gc


def prepare_dataset(dataset):
    # LayerPreprocesses adds self loops and normalizes the adjacency matrix
    dataset = Citation(
        dataset,
        normalize_x=True,
        transforms=[LayerPreprocess(GCNConv)],
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


def run_experiment(
    experiment_name,
    num_runs,
    epochs,
    dataset,
    model_factory,
    optimizer,
    loss_function,
    callbacks,
    reduce_memory_overhead=True,
    verbose=0,
):
    loader_tr, loader_va, loader_te = generate_data_loaders(dataset)

    history_dataframes = []
    eval_data = []
    for run in trange(num_runs):
        model = model_factory()

        model.compile(
            optimizer=optimizer, loss=loss_function, weighted_metrics=["accuracy"]
        )

        history = model.fit(
            loader_tr.load(),
            steps_per_epoch=loader_tr.steps_per_epoch,
            validation_data=loader_va.load(),
            validation_steps=loader_va.steps_per_epoch,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose,
        )
        eval_results = model.evaluate(
            loader_te.load(), steps=loader_te.steps_per_epoch, verbose=verbose
        )

        # Do this to reduce memory overhead since we are using dense tensors
        if reduce_memory_overhead:
            del model
            gc.collect()

        history_df = pd.DataFrame(history.history)
        history_df["run"] = run
        history_dataframes.append(history_df)

        eval_data.append(eval_results)

    history_df = pd.concat(history_dataframes)
    eval_df = pd.DataFrame(eval_data).reset_index()
    eval_df.columns = ["run", "test_loss", "test_accuracy"]

    history_df.to_csv(f"./data/{experiment_name}_{dataset.name}_history.csv")
    eval_df.to_csv(f"./data/{experiment_name}_{dataset.name}_eval.csv")

    mean_train_acc = history_df.groupby("run")["accuracy"].max().mean()
    mean_val_acc = history_df.groupby("run")["val_accuracy"].max().mean()
    mean_test_acc = eval_df["test_accuracy"].mean()
    print(
        f"{num_runs} runs completed: {mean_train_acc:.4f} mean train acc, {mean_val_acc:.4f} mean val acc, {mean_test_acc:.4f} mean test acc"
    )
