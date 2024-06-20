"""
Script for training models for semantic segmentation
Author: Yanjun Wu (github/yaw6622)
License: MIT
"""
import argparse
import json
import os
import pprint
import time
import pickle as pkl

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchnet as tnt
from Function.CF_Dataset import CFDataset
from Function.metrics import confusion_matrix_analysis
from Function.miou import IoU
from Function import utils
from Function.weight_init import weight_init
import load_model
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    default="JM",
    type=str,
    help="Which dataset to use. Can be one of: (JM/CF/PAS_SS)",
)

## Hyperparameters of DBL
# parser.add_argument("--encoder_widths", default="[64,64,64,128]", type=str)
# parser.add_argument("--decoder_widths", default="[32,32,64,128]", type=str)
parser.add_argument("--out_conv", default="[32, 3]")
# parser.add_argument("--str_conv_k", default=4, type=int)
# parser.add_argument("--str_conv_s", default=2, type=int)
# parser.add_argument("--str_conv_p", default=1, type=int)
# parser.add_argument("--agg_mode", default="att_group", type=str)
# parser.add_argument("--encoder_norm", default="group", type=str)
# parser.add_argument("--n_head", default=16, type=int)
# parser.add_argument("--d_model", default=256, type=int)
# parser.add_argument("--d_k", default=4, type=int)

# Set-up parameters
parser.add_argument(
    "--dataset_folder",
    default="",
    type=str,
    help="Path to the dataset folder.",
)
parser.add_argument(
    "--res_dir",
    default="./results",
    help="Path to the folder where the results should be stored",
)
parser.add_argument(
    "--num_workers", default=8, type=int, help="Number of workers"
)
parser.add_argument("--rdm_seed", default=1, type=int, help="Random seed")
parser.add_argument(
    "--device",
    default="cuda",
    type=str,
    help="Name of device to use for tensor computations (cuda/cpu)",
)
parser.add_argument(
    "--display_step",
    default=50,
    type=int,
    help="Interval in batches between display of training metrics",
)
parser.add_argument(
    "--cache",
    dest="cache",
    action="store_true",
    help="If specified, the whole dataset is kept in RAM",
)
# Training parameters
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs per fold")
parser.add_argument("--batch_size", default=2, type=int, help="Batch size")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
# parser.add_argument("--mono_date", default=None, type=str)
# parser.add_argument("--ref_date", default="2018-09-01", type=str)

parser.add_argument("--num_classes", default=3, type=int)
parser.add_argument("--ignore_index", default=None)
parser.add_argument("--pad_value", default=0, type=float)
parser.add_argument("--padding_mode", default="reflect", type=str)
parser.add_argument(
    "--val_every",
    default=1,
    type=int,
    help="Interval in epochs between two validation steps.",
)
parser.add_argument(
    "--val_after",
    default=0,
    type=int,
    help="Do validation only after that many epochs.",
)

list_args = ["encoder_widths", "decoder_widths", "out_conv"]
parser.set_defaults(cache=False)

"""
The implementations of functions iterate, recursive_todevice, prepare_output, checkpoint, save_results, overall_performance 
are based on the code of the project 'Utae' (Author: Vivien Sainte Fare Garnot (github/VSainteuf)).
These functions have been modified and adjusted from their original code.
"""
def iterate(
    model, data_loader, criterion, config, optimizer=None, mode="train", device=None,
):
    loss_meter = tnt.meter.AverageValueMeter()
    # 为单值数据取平均及方差计算
    iou_meter = IoU(
        num_classes=config.num_classes,
        ignore_index=config.ignore_index,
        cm_device=config.device,
    )

    t_start = time.time()
    for i, batch in enumerate(data_loader):
        if device is not None:
            batch = recursive_todevice(batch, device)
        x,  y = batch
        x = x.float()
        y = y.long()

        if mode != "train":
            with torch.no_grad():
                out = model(x)
        else:
            optimizer.zero_grad()
            out = model(x)

        loss = criterion(out, y)

        if mode == "train":
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            pred = out.argmax(dim=1)   #[4,128,128]
        iou_meter.add(pred, y)
        loss_meter.add(loss.item())

        if (i + 1) % config.display_step == 0:
            miou, acc = iou_meter.get_miou_acc()
            print(
                "Step [{}/{}], Loss: {:.4f}, Acc : {:.2f}, mIoU {:.2f}".format(
                    i + 1, len(data_loader), loss_meter.value()[0], acc, miou
                )
            )

    t_end = time.time()
    total_time = t_end - t_start
    print("Epoch time : {:.1f}s".format(total_time))
    miou, acc = iou_meter.get_miou_acc()
    metrics = {
        "{}_accuracy".format(mode): acc,
        "{}_loss".format(mode): loss_meter.value()[0],
        "{}_IoU".format(mode): miou,
        "{}_epoch_time".format(mode): total_time,
    }

    if mode == "test":
        return metrics, iou_meter.conf_metric.value()  # confusion matrix
    else:
        return metrics

def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: recursive_todevice(v, device) for k, v in x.items()}
    else:
        return [recursive_todevice(c, device) for c in x]


def prepare_output(config):
    os.makedirs(config.res_dir, exist_ok=True)

def checkpoint(log, config):
    with open(
        os.path.join(config.res_dir, "trainlog.json"), "w"
    ) as outfile:
        json.dump(log, outfile, indent=4)


def save_results(metrics, conf_mat, config):
    with open(
        os.path.join(config.res_dir, "test_metrics.json"), "w"
    ) as outfile:
        json.dump(metrics, outfile, indent=4)
    pkl.dump(
        conf_mat,
        open(
            os.path.join(config.res_dir, "conf_mat.pkl"), "wb"
        ),
    )

def overall_performance(config):
    cm = np.zeros((config.num_classes, config.num_classes))
    with open(os.path.join(config.res_dir, "conf_mat.pkl"), "rb") as f:
        cm = pkl.load(f)


    # if config.ignore_index is not None:
    #     cm = np.delete(cm, config.ignore_index, axis=0)
    #     cm = np.delete(cm, config.ignore_index, axis=1)

    _, perf = confusion_matrix_analysis(cm)


    print("Overall performance:")
    print("Acc: {},  IoU: {}".format(perf["Accuracy"], perf["MACRO_IoU"]))
    perf_converted = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in perf.items()}

    with open(os.path.join(config.res_dir, "overall.json"), "w") as file:
        file.write(json.dumps(perf_converted, indent=4))

def main(config):

    prepare_output(config)
    device = torch.device(config.device)
    if config.model == "JM":
        dt_train = CFDataset(data_dir='JINGMEN/Jingmen_Data/Jingmen_Train',
                         label_dir='JINGMEN/Jingmen_Label/Label_Train',
                         norm=True) #dir to the JM train set
        dt_val = CFDataset(data_dir='JINGMEN/Jingmen_Data/Jingmen_Val',
                       label_dir='JINGMEN/Jingmen_Label/Label_Val', norm=True) #dir to the JM val set
        dt_test = CFDataset(data_dir='JINGMEN/Jingmen_Data/Jingmen_Test',
                        label_dir='JINGMEN/Jingmen_Label/Label_Test', norm=True) #dir to the JM test set
    elif config.model == "CF":
        dt_train = CFDataset(data_dir='CFDATA/mydata/CF_Train',
                          label_dir='CFlabel/mylabel/Label_Train',
                          norm=True) #dir to the CF train set
        dt_val = CFDataset(data_dir='CFDATA/mydata/CF_Val',
                        label_dir='CFlabel/mylabel/Label_Val', norm=True) #dir to the CF val set
        dt_test = CFDataset(data_dir='CFDATA/mydata/CF_Test',
                        label_dir='CFlabel/mylabel/Label_Test', norm=True) #dir to the CF test set

    collate_fn = lambda x: utils.pad_collate(x, pad_value=config.pad_value)

    train_loader = data.DataLoader(
        dt_train,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    val_loader = data.DataLoader(
        dt_val,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    test_loader = data.DataLoader(
        dt_test,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn
    )

    print(
        "Train {}, Val {}, Test {}".format(len(dt_train), len(dt_val), len(dt_test))
    )
    #load DBL model
    model = load_model.get_model(config, mode="semantic")
    config.N_params = utils.get_ntrainparams(model)
    with open(os.path.join(config.res_dir, "conf.json"), "w") as file:
        file.write(json.dumps(vars(config), indent=4))
    #print(model)
    print("TOTAL TRAINABLE PARAMETERS :", config.N_params)
    print("Trainable layers:")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(name)
    model = model.to(device)
    model.apply(weight_init)
    #load saved model if needed

    # sd = torch.load(
    #     os.path.join('results', "model.pth.tar"),
    #     map_location=device,
    # )
    # model.load_state_dict(sd["state_dict"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    weights = torch.ones(config.num_classes, device=device).float()
    criterion = nn.CrossEntropyLoss(weight=weights)
    trainlog = {}
    best_mIoU = 0
    for epoch in range(1, config.epochs + 1):
        print("EPOCH {}/{}".format(epoch, config.epochs))

        model.train()
        train_metrics = iterate(
            model,
            data_loader=train_loader,
            criterion=criterion,
            config=config,
            optimizer=optimizer,
            mode="train",
            device=device,
            # epoch=epoch,
        )
        if epoch % config.val_every == 0 and epoch > config.val_after:
            print("Validation . . . ")
            model.eval()
            val_metrics = iterate(
                model,
                data_loader=val_loader,
                criterion=criterion,
                config=config,
                optimizer=optimizer,
                mode="val",
                device=device,
                # epoch=epoch,
            )

            print(
                "Loss {:.4f},  Acc {:.2f},  IoU {:.4f}".format(
                    val_metrics["val_loss"],
                    val_metrics["val_accuracy"],
                    val_metrics["val_IoU"],
                )
            )

            trainlog[epoch] = {**train_metrics, **val_metrics}
            checkpoint(trainlog, config)
            if val_metrics["val_IoU"] >= best_mIoU:
                best_mIoU = val_metrics["val_IoU"]
                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    os.path.join(
                        config.res_dir,  "model.pth.tar"
                    ),
                )
        else:
            trainlog[epoch] = {**train_metrics}
            checkpoint(trainlog, config)

    print("Testing best epoch . . .")
    model.load_state_dict(
        torch.load(
            os.path.join(
                config.res_dir, "model.pth.tar"
            )
        )["state_dict"]
    )
    model.eval()

    test_metrics, conf_mat = iterate(
        model,
        data_loader=test_loader,
        criterion=criterion,
        config=config,
        optimizer=optimizer,
        mode="test",
        device=device,
        # epoch=epoch,
    )
    print(
        "Loss {:.4f},  Acc {:.2f},  IoU {:.4f}".format(
            test_metrics["test_loss"],
            test_metrics["test_accuracy"],
            test_metrics["test_IoU"],
        )
    )
    save_results(test_metrics, conf_mat.cpu().numpy(), config)

    overall_performance(config)

if __name__ == "__main__":
    config = parser.parse_args()
    # config.dataset_folder="./PASTIS"

    for k, v in vars(config).items():
        if k in list_args and v is not None:
            v = v.replace("[", "")
            v = v.replace("]", "")
            config.__setattr__(k, list(map(int, v.split(","))))

    assert config.num_classes == config.out_conv[-1]

    pprint.pprint(config)
    main(config)


