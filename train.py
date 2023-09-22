
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
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss
# from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

data = pd.read_csv("data/network/network_data_15min_processed.csv")
data["line"] = data["line"].astype('category').astype(str)
print("*" * 50, data.columns)
print(data.dtypes)

max_prediction_length = 288
max_encoder_length = 1440
training_cutoff = data["time_idx"].max() - max_prediction_length

training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="in",
    group_ids=["customer", "line"],
    min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["customer", "line"],
    time_varying_known_categoricals=[],
    time_varying_known_reals=["month", "day", "hour", "minute"],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=["out"],
    target_normalizer=GroupNormalizer(
        groups=["customer", "line"], transformation="softplus"
    ),  # use softplus and normalize by group
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

# create validation set (predict=True) which means to predict the last max_prediction_length points in time
# for each series
validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)

def train(train_dataloader=None, val_dataloader=None, gpu=True, batch_limit=1.0):
    # configure network and trainer
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
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
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=8,
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
    )

    best_model_path = trainer.checkpoint_callback.best_model_path
    print("`"*50, "best model path", best_model_path)
    return best_model_path


def test(best_model_path, val_dataloader):
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    predictions = best_tft.predict(val_dataloader, return_y=True, trainer_kwargs=dict(accelerator="cuda"))
    print("MAE", MAE()(predictions.output, predictions.y))

    # calculate baseline mean absolute error, i.e. predict next value as the last available value from the history
    baseline_predictions = Baseline().predict(val_dataloader, return_y=True)
    print("Baseline", MAE()(baseline_predictions.output, baseline_predictions.y))

    # # raw predictions are a dictionary from which all kind of information including quantiles can be extracted
    raw_predictions = best_tft.predict(val_dataloader, mode="raw", return_x=True)
    print("*"*50, raw_predictions.x["groups"])
    print("~"*50, raw_predictions.x)
    print("~"*50, raw_predictions.y)
    fig, ax = plt.subplots(4, 2)
    ax = ax.ravel()
    for idx in range(8):  # plot 10 examples
        best_tft.plot_prediction(raw_predictions.x, raw_predictions.output, idx=idx, add_loss_to_title=True, ax=ax[idx])
    ax[0].legend()
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")
    parser.add_argument("--bs", type=int, default=128, help="batch size")
    parser.add_argument("--bl", type=float, default=1.0, help="batch limit")

    # opt = parser.parse_args()
    opt = parser.parse_args(args=[])  # for Jupyter Notebook (Google Colab / Kaggle)
    # create dataloaders for model
    batch_size = opt.bs  # set this between 32 to 128
    # print(training)
    print("#"*60)
    print(f"##  USE GPU={opt.gpu} | Batch Size={opt.bs} | Batch Limit={opt.bl}")
    print("#"*60)
    # train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    # val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)
    #
    # best_model_path = train(train_dataloader=train_dataloader, val_dataloader=val_dataloader,
    #                         gpu=opt.gpu, batch_limit=opt.bl)
    # print("best model path", best_model_path)
    # test(best_model_path, val_dataloader)
    # test("lightning_logs/lightning_logs/version_19/checkpoints/epoch=30-step=3100.ckpt")
