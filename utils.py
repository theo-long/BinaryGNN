import tensorflow as tf

from spektral.datasets import Citation
from spektral.transforms import LayerPreprocess, AdjToSpTensor
from spektral.layers import GCNConv
from spektral.data import SingleLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange
import gc
import glob
import re
import os


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
    remove_old_model_files=True,
    recreate_optimizer=False,
):
    loader_tr, loader_va, loader_te = generate_data_loaders(dataset)

    history_dataframes = []
    eval_data = []

    # Clear out old model files
    if remove_old_model_files:
        model_files = glob.glob("model_files/*")
        for f in model_files:
            os.remove(f)

    for run in trange(num_runs):
        if recreate_optimizer:
            opt = optimizer()
        else:
            opt = optimizer

        checkpoint_path = f"model_files/{experiment_name}_run_{run}.tf"
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
        )

        model: tf.keras.Model = model_factory()

        model.compile(optimizer=opt, loss=loss_function, weighted_metrics=["accuracy"])

        history = model.fit(
            loader_tr.load(),
            steps_per_epoch=loader_tr.steps_per_epoch,
            validation_data=loader_va.load(),
            validation_steps=loader_va.steps_per_epoch,
            epochs=epochs,
            callbacks=callbacks + [checkpoint_callback],
            verbose=verbose,
        )
        # Restore best weights
        model.load_weights(checkpoint_path)
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


def plot_training_curves(
    history_df,
    hyperparameter_name,
    error_bars=True,
    acc_plot_kwargs=None,
    loss_plot_kwargs=None,
):
    if not acc_plot_kwargs:
        acc_plot_kwargs = {}
    if not loss_plot_kwargs:
        loss_plot_kwargs = {}

    median_df = history_df.groupby(["epoch", hyperparameter_name])[
        ["loss", "accuracy", "val_loss", "val_accuracy"]
    ].quantile(0.5)
    top_quantile_df = history_df.groupby(["epoch", hyperparameter_name])[
        ["loss", "accuracy", "val_loss", "val_accuracy"]
    ].quantile(0.75)
    bottom_quantile_df = history_df.groupby(["epoch", hyperparameter_name])[
        ["loss", "accuracy", "val_loss", "val_accuracy"]
    ].quantile(0.25)

    try:
        hyper_values = history_df[hyperparameter_name].sort_values().unique()
    except Exception:
        hyper_values = history_df[hyperparameter_name].unique()

    plots = []
    for metric in ["accuracy", "loss"]:
        for prefix in ["", "val_"]:
            plot_kwargs = acc_plot_kwargs if metric == "accuracy" else loss_plot_kwargs
            ax = median_df[prefix + metric].unstack().plot(**plot_kwargs)
            if error_bars:
                lower = bottom_quantile_df[prefix + metric].unstack()
                upper = top_quantile_df[prefix + metric].unstack()
                for hyper_value in hyper_values:
                    plt.fill_between(
                        lower.index, lower[hyper_value], upper[hyper_value], alpha=0.4
                    )

            ax.set_title(f"{prefix.strip('_').title()} {metric.capitalize()}")
            plots.append(ax)

    return plots


def generate_dataframes_from_files(file_string, param_column=None):
    eval_dfs = []
    history_dfs = []
    for f in glob.glob(f"{file_string}*eval*"):
        df = pd.read_csv(f)
        if param_column:
            df[param_column] = float(re.match(".*=(.*)_c", f).groups()[0])
        eval_dfs.append(df)
    for f in glob.glob(f"{file_string}*history*"):
        df = pd.read_csv(f)
        if param_column:
            df[param_column] = float(re.match(".*=(.*)_c", f).groups()[0])
        history_dfs.append(df)
    eval_df = pd.concat(eval_dfs)
    history_df = pd.concat(history_dfs)
    history_df = history_df.rename(columns={"Unnamed: 0": "epoch"})
    return eval_df, history_df
