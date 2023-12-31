
import os
import warnings
import matplotlib.pyplot as plt
import tensorboard

warnings.filterwarnings("ignore")  # avoid printing out absolute paths

import copy
from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import torch
import torch.nn as nn

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss
# from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from lightning.pytorch.tuner import Tuner


interval = 30
target_date = "2023-10-02"
target_date = pd.to_datetime(target_date)
data = pd.read_csv(f"data/network/network_data_{interval}min-1yr_processed.csv")
data["line"] = data["line"].astype('category').astype(str)
data["time"] = pd.to_datetime(data["time"])
data = data[data["time"] <= target_date]
# print("*" * 50)
# print(data.columns)
# print(data.dtypes)
# print("NA\n", data[data["log_in"].isna()])

max_prediction_length = int(24 * 60 / interval) * 2    # One day
max_encoder_length = int(max_prediction_length * 7)  # One week
training_cutoff = data["time_idx"].max() - max_prediction_length

print("time idx ", data["time_idx"].max())
print("prediction length", max_prediction_length)
print("encoder length", max_encoder_length)

training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="in_log",
    group_ids=["customer", "line"],
    min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["customer", "line", "is_holiday"],
    time_varying_known_categoricals=[],
    time_varying_known_reals=["month_cos", "weekday_cos", "day_cos", "hour_cos", "minute_cos"],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=["in_log", "in_log_avg", "in_log_var"],
    target_normalizer=GroupNormalizer(
        groups=["customer", "line"], transformation='softplus'
    ),  # use softplus and normalize by group
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

# print("train dataset\n", training.index)
print(f"training target <{training.target}>")
# create validation set (predict=True) which means to predict the last max_prediction_length points in time
# for each series
validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)
# print("validation dataset\n", validation.index)


def train(train_dataloader=None, val_dataloader=None, gpu=True, batch_limit=1.0, restore_path=None):
    # configure network and trainer
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-5, patience=10, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()  # log the learning rate
    logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

    trainer = pl.Trainer(
        max_epochs=60,
        accelerator="cuda" if gpu else "cpu",
        enable_model_summary=True,
        gradient_clip_val=0.1,
        limit_train_batches=batch_limit,  # coment in for training, running valiation every 30 batches
        # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
        callbacks=[lr_logger, early_stop_callback],
        logger=logger,
    )

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.001,
        hidden_size=80,
        attention_head_size=8,
        dropout=0.1,
        hidden_continuous_size=80,
        loss=QuantileLoss(),
        log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
        optimizer="Ranger",
        reduce_on_plateau_patience=4,
    )
    print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

    # fit network
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=restore_path  # None=start from fresh, path=start from restore path
    )

    best_model_path = trainer.checkpoint_callback.best_model_path
    print("`"*50, "\n", "best model path: ", best_model_path)
    return best_model_path


def find_lr(train_dataloader=None, val_dataloader=None, gpu=True, batch_limit=1.0):
    # configure network and trainer
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-5, patience=10, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()  # log the learning rate
    logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

    trainer = pl.Trainer(
        max_epochs=60,
        accelerator="cuda" if gpu else "cpu",
        enable_model_summary=True,
        gradient_clip_val=0.1,
        limit_train_batches=batch_limit,  # coment in for training, running valiation every 30 batches
        # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
        callbacks=[lr_logger, early_stop_callback],
        logger=logger,
    )

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.001,
        hidden_size=320,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=160,
        loss=QuantileLoss(),
        log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
        optimizer="Ranger",
        reduce_on_plateau_patience=4,
    )
    print(f"Number of parameters in network: {tft.size() / 1e3:.1f}k")
    res = Tuner(trainer).lr_find(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        max_lr=10.0,
        min_lr=1e-6,
    )

    print(f"suggested learning rate: {res.suggestion()}")
    fig = res.plot(show=True, suggest=True)
    fig.show()


def test(best_model_path, val_dataloader):
    map_location = None if torch.cuda.is_available() else torch.device('cpu')
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path, map_location=map_location)
    predictions = best_tft.predict(val_dataloader, return_y=True, trainer_kwargs=dict(accelerator="cuda"))

    for batch in val_dataloader:
        data, labels = batch
        # for key, value in data.items():
        #     print(key, value.shape)
        # print(labels)
        print("target scale\n", data["target_scale"])
        print("decoder target\n", data["decoder_target"].size())
        for l in labels:
            if l is not None:
                print("labels:   ", l.shape)
            else:
                print("labels:    None.")

    print("MAE", MAE()(predictions.output, predictions.y))

    # calculate baseline mean absolute error, i.e. predict next value as the last available value from the history
    baseline_predictions = Baseline().predict(val_dataloader, return_y=True)
    print("Baseline", MAE()(baseline_predictions.output, baseline_predictions.y))

    # # # raw predictions are a dictionary from which all kind of information including quantiles can be extracted
    raw_predictions = best_tft.predict(val_dataloader, mode="raw", return_x=True, return_y=True)

    print(raw_predictions.x["target_scale"])

    fig, ax = plt.subplots(2, 2)
    ax = ax.ravel()

    print(raw_predictions.x.keys())

    file_output = "output_for_plot.csv"
    if os.path.exists(file_output):
        data_write = pd.read_csv(file_output)
        data_write = data_write.to_dict(orient='list')
    else:
        data_write = {"time": [],
                      "line": [],
                      "gt": [],
                      "0.02": [], "0.10": [], "0.25": [], "0.5": [], "0.75": [], "0.90": [], "0.98": []}
    minimum_time = min(data_write["time"]) if data_write["time"] != [] else 0

    for idx in range(4):  # plot 10 examples
        # print("ax", idx)
        """
        NOTE for this function
        1. y_raw is actually equal to y_quantile. Their shape will be [num_of_groups, prediction_length, num_of_quantiles]
        2. y_all is actually equal to y. Why it needs another torch.cat?
        3. y_hat is the prediction.
        4. output["prediction"] is y_raw
        """
        best_tft.plot_prediction(raw_predictions.x, raw_predictions.output, idx=idx, add_loss_to_title=True, ax=ax[idx])
        print("raw pred pred", raw_predictions.output["prediction"].shape, raw_predictions.output["prediction"][idx, 0, :])
        print("pred xxxxx", raw_predictions.x.keys())
        print("pred output", raw_predictions.output.keys())
        print("x decoder target ++++++++++++++", raw_predictions.x["decoder_target"][idx, :].tolist())

        for ii, qi in enumerate(["0.02", "0.10", "0.25", "0.5", "0.75", "0.90", "0.98"]):
            data_write[qi] += raw_predictions.output["prediction"][idx, :, ii].tolist()
        data_write["time"] += list(range(minimum_time - max_prediction_length, minimum_time))
        data_write["gt"] += raw_predictions.x["decoder_target"][idx, :].tolist()

        # best_tft.plot_prediction(raw_predictions.x, raw_predictions.output, idx=idx, add_loss_to_title=False, ax=ax[idx])
        v = raw_predictions.x["groups"][idx]
        # print(next(key for key, value in best_tft.hparams["embedding_labels"]["customer"] if value == v[0]))
        subtitle = (next(key for key, value in best_tft.hparams["embedding_labels"]["customer"].items() if value == v[0]) +
                    ", " +
                    next(key for key, value in best_tft.hparams["embedding_labels"]["line"].items() if value == v[1]))
        print("subtitle", subtitle)
        data_write["line"] += [subtitle] * max_prediction_length
        # ax[idx].set_title(subtitle)

    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    # predictions = best_tft.predict(val_dataloader, return_x=True)
    # print("~~~" * 3, predictions.x)
    # predictions_vs_actuals = best_tft.calculate_prediction_actual_by_variable(predictions.x, predictions.output)
    # best_tft.plot_prediction_actual_by_variable(predictions_vs_actuals)

    interpretation = best_tft.interpret_output(raw_predictions.output, reduction="sum")
    best_tft.plot_interpretation(interpretation)

    # print("``````````````````````` \n", data_write)
    for key, value in data_write.items():
        print(key, len(value))
    data_write = pd.DataFrame(data_write)
    data_write.sort_values(by=["line", "time"], ascending=[True, True])
    data_write.to_csv("output_for_plot.csv", index=False)

    # print("@"*70, "\n", best_tft.hparams)
    plt.show()



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")
    parser.add_argument("--bs", type=int, default=64, help="batch size")
    parser.add_argument("--bl", type=float, default=1.0, help="batch limit")
    parser.add_argument("--restore", type=str, default=None, help="restore ckpt path")

    opt = parser.parse_args()
    # opt = parser.parse_args(args=[])  # for Jupyter Notebook (Google Colab / Kaggle)
    # create dataloaders for model
    batch_size = opt.bs  # set this between 32 to 128
    # print(training)
    print("#"*60)
    print(f"##  USE GPU={opt.gpu} | Batch Size={opt.bs} | Batch Limit={opt.bl}")
    print("#"*60)
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

    # best_model_path = train(train_dataloader=train_dataloader, val_dataloader=val_dataloader,
    #                         gpu=opt.gpu, batch_limit=opt.bl, restore_path=opt.restore)
    # test(best_model_path, val_dataloader)
    test("lightning_logs/lightning_logs/colab_1016_oneyear_30min/checkpoints/epoch=56-step=10317.ckpt", val_dataloader)
    # test("lightning_logs/lightning_logs/colab_1013_inlognorm/checkpoints/epoch=27-step=4564.ckpt", val_dataloader)
    # test("lightning_logs/lightning_logs/colab_1012_hsize=80_csize=80_atthead=8/checkpoints/epoch=59-step=9780.ckpt", val_dataloader)

    # find_lr(train_dataloader=train_dataloader, val_dataloader=val_dataloader, gpu=True, batch_limit=opt.bl)

