# -*- coding: utf-8 -*-

import os

# 必须位于 transformers 导入之前
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import cv2
import torch
import numpy as np

from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader

from transformers import SegformerForSemanticSegmentation

import matplotlib.pyplot as plt

# =====================================================
# Paths
# =====================================================


YOLO_ROOT = (
r"G:\water_environment\Urban_Waterlogging_Detection"
r"\Baselines\6. YOLOv8-seg\data\UW-Bench_YOLO"
)


IMG_TRAIN = os.path.join(
    YOLO_ROOT,
    "images/train"
)


IMG_VAL = os.path.join(
    YOLO_ROOT,
    "images/val"
)


MASK_DIR = (
r"G:\water_environment\Urban_Waterlogging_Detection"
r"\data\UW-Bench\training_set\SegmentationClass"
)



SAVE_DIR="segformer_checkpoints"


os.makedirs(
    SAVE_DIR,
    exist_ok=True
)



# =====================================================
# Dataset
# =====================================================


class WaterDataset(Dataset):

    def __init__(
            self,
            img_dir,
            mask_dir
    ):

        self.img_dir=img_dir

        self.mask_dir=mask_dir

        self.images=sorted(
            os.listdir(img_dir)
        )



    def __len__(self):

        return len(self.images)



    def __getitem__(self,index):

        name=self.images[index]


        img_path=os.path.join(
            self.img_dir,
            name
        )


        mask_path=os.path.join(
            self.mask_dir,
            os.path.splitext(name)[0]+".png"
        )


        # image

        img=cv2.imread(
            img_path
        )


        img=cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )


        img=cv2.resize(
            img,
            (512,512)
        )


        img=img/255.



        img=torch.tensor(
            img,
            dtype=torch.float32
        ).permute(
            2,0,1
        )



        # mask

        mask=cv2.imread(
            mask_path,
            0
        )


        mask=cv2.resize(
            mask,
            (512,512),
            interpolation=cv2.INTER_NEAREST
        )


        mask=(
            mask>0
        ).astype(
            np.int64
        )



        mask=torch.tensor(
            mask,
            dtype=torch.long
        )


        return img,mask




# =====================================================
# Dice Loss
# =====================================================


class DiceLoss(nn.Module):


    def forward(
            self,
            pred,
            target
    ):


        pred=torch.softmax(
            pred,
            dim=1
        )


        target_onehot=torch.nn.functional.one_hot(
            target,
            num_classes=2
        )


        target_onehot=target_onehot.permute(
            0,3,1,2
        ).float()



        smooth=1



        intersection=(
            pred*target_onehot
        ).sum(
            dim=(2,3)
        )


        union=(
            pred+
            target_onehot
        ).sum(
            dim=(2,3)
        )


        dice=(
            (2*intersection+smooth)
            /
            (union+smooth)
        )


        return 1-dice.mean()



# =====================================================
# Loss
# =====================================================


class ComboLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.ce=nn.CrossEntropyLoss()

        self.dice=DiceLoss()



    def forward(
            self,
            pred,
            target
    ):

        return (
            self.ce(pred,target)
            +
            self.dice(pred,target)
        )



# =====================================================
# Model
# =====================================================
LOCAL_MODEL_DIR = r"G:\water_environment\Urban_Waterlogging_Detection\Baselines\4. SegFormer-B0\segformer-b0-finetuned-ade-512-512"


def get_model():

    print("=" * 60)
    print("LOCAL_MODEL_DIR:")
    print(LOCAL_MODEL_DIR)
    print("目录存在:", os.path.isdir(LOCAL_MODEL_DIR))
    print("config存在:",
          os.path.isfile(os.path.join(LOCAL_MODEL_DIR, "config.json")))
    print("safetensors存在:",
          os.path.isfile(os.path.join(LOCAL_MODEL_DIR, "model.safetensors")))
    print("bin存在:",
          os.path.isfile(os.path.join(LOCAL_MODEL_DIR, "pytorch_model.bin")))
    print("=" * 60)

    model = SegformerForSemanticSegmentation.from_pretrained(
        LOCAL_MODEL_DIR,
        num_labels=2,
        id2label={
            0: "background",
            1: "water"
        },
        label2id={
            "background": 0,
            "water": 1
        },
        ignore_mismatched_sizes=True,
        local_files_only=True
    )

    return model



# =====================================================
# Train
# =====================================================


def train():


    device=torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )



    print("="*60)

    print(
        "Device:",
        device
    )

    print(
        "Train images:",
        len(os.listdir(IMG_TRAIN))
    )

    print("="*60)



    dataset=WaterDataset(
        IMG_TRAIN,
        MASK_DIR
    )



    loader=DataLoader(

        dataset,

        batch_size=8,

        shuffle=True,

        num_workers=0

    )



    model=get_model()

    model.to(device)



    criterion=ComboLoss()



    optimizer=torch.optim.AdamW(

        model.parameters(),

        lr=6e-5,

        weight_decay=0.01

    )



    scaler=torch.cuda.amp.GradScaler()



    epochs=150


    loss_history=[]



    for epoch in range(epochs):


        model.train()


        total_loss=0



        pbar=tqdm(

            loader,

            desc=f"Epoch [{epoch+1}/{epochs}]"

        )



        for img,mask in pbar:


            img=img.to(device)

            mask=mask.to(device)



            with torch.cuda.amp.autocast():


                output=model(
                    pixel_values=img
                )


                logits=output.logits



                # resize prediction

                logits=torch.nn.functional.interpolate(

                    logits,

                    size=mask.shape[-2:],

                    mode="bilinear",

                    align_corners=False

                )


                loss=criterion(
                    logits,
                    mask
                )



            optimizer.zero_grad()


            scaler.scale(
                loss
            ).backward()


            scaler.step(
                optimizer
            )


            scaler.update()



            total_loss+=loss.item()



            pbar.set_postfix(
                loss=loss.item()
            )



        epoch_loss=(
            total_loss/
            len(loader)
        )


        loss_history.append(
            epoch_loss
        )


        print(
            f"Epoch {epoch+1}/{epochs} Loss:{epoch_loss:.4f}"
        )



        torch.save(

            model.state_dict(),

            f"{SAVE_DIR}/segformer_b0_epoch_{epoch+1}.pth"

        )



    # loss curve


    plt.figure()

    plt.plot(
        loss_history
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Loss"
    )


    plt.title(
        "SegFormer-B0 Training Loss"
    )


    plt.grid()


    plt.savefig(
        "segformer_loss_curve.png",
        dpi=300
    )




if __name__=="__main__":

    train()