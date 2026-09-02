# -*- coding: utf-8 -*-

import os

# 离线模式
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


import cv2
import time
import torch
import numpy as np
import pandas as pd


from transformers import SegformerForSemanticSegmentation



# =====================================================
# Path
# =====================================================


YOLO_ROOT = (
r"G:\water_environment\Urban_Waterlogging_Detection"
r"\Baselines\6. YOLOv8-seg\data\UW-Bench_YOLO"
)


VAL_IMG = os.path.join(
    YOLO_ROOT,
    "images/val"
)



GT_DIR = (
r"G:\water_environment\Urban_Waterlogging_Detection"
r"\data\UW-Bench\training_set\SegmentationClass"
)



# SegFormer 原始模型目录

LOCAL_MODEL_DIR = (
r"G:\water_environment\Urban_Waterlogging_Detection"
r"\Baselines\4. SegFormer-B0"
r"\segformer-b0-finetuned-ade-512-512"
)



# 训练保存权重

MODEL_PATH = (
r"segformer_checkpoints"
r"\segformer_b0_epoch_150.pth"
)



SAVE_DIR = (
r"runs\segformer_val_result"
)



CSV_PATH = (
r"runs\segformer_metrics.csv"
)



os.makedirs(
    SAVE_DIR,
    exist_ok=True
)




# =====================================================
# Model
# =====================================================


def get_model():


    model = SegformerForSemanticSegmentation.from_pretrained(
        LOCAL_MODEL_DIR,
        num_labels=2,
        id2label={
            0:"background",
            1:"water"
        },
        label2id={
            "background":0,
            "water":1
        },
        ignore_mismatched_sizes=True,
        local_files_only=True
    )


    return model




# =====================================================
# Load
# =====================================================


device=torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)



print("="*60)

print("SegFormer-B0 Evaluation")

print(
    "Device:",
    device
)



model=get_model()



checkpoint=torch.load(
    MODEL_PATH,
    map_location=device
)



model.load_state_dict(
    checkpoint,
    strict=True
)



model.to(device)

model.eval()



# =====================================================
# Params
# =====================================================


params=sum(
    p.numel()
    for p in model.parameters()
)


print(
    f"Params(M)       : {params/1e6:.2f}"
)




# =====================================================
# Metrics
# =====================================================


TP=0
FP=0
FN=0
TN=0



records=[]



total_time=0

num_images=0




# =====================================================
# Test
# =====================================================


for name in sorted(os.listdir(VAL_IMG)):


    img_path=os.path.join(
        VAL_IMG,
        name
    )


    gt_path=os.path.join(
        GT_DIR,
        os.path.splitext(name)[0]+".png"
    )


    if not os.path.exists(gt_path):

        continue



    # =========================
    # Image
    # =========================


    img=cv2.imread(
        img_path
    )


    rgb=cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )


    rgb=cv2.resize(
        rgb,
        (512,512)
    )


    rgb=rgb.astype(
        np.float32
    )/255.



    tensor=torch.tensor(
        rgb,
        dtype=torch.float32
    )


    tensor=tensor.permute(
        2,0,1
    )


    tensor=tensor.unsqueeze(
        0
    )


    tensor=tensor.to(device)




    # =========================
    # Inference
    # =========================


    if device.type=="cuda":

        torch.cuda.synchronize()



    start=time.time()



    with torch.no_grad():


        output=model(
            pixel_values=tensor
        )


        logits=output.logits



        # SegFormer输出1/4尺寸

        logits=torch.nn.functional.interpolate(
            logits,
            size=(512,512),
            mode="bilinear",
            align_corners=False
        )



    if device.type=="cuda":

        torch.cuda.synchronize()



    total_time += (
        time.time()-start
    )


    num_images += 1




    # =========================
    # Prediction
    # =========================


    pred=torch.argmax(
        logits,
        dim=1
    )


    pred=pred.squeeze(
        0
    ).cpu().numpy()



    pred=(
        pred>0
    ).astype(
        np.uint8
    )




    # =========================
    # Ground Truth
    # =========================


    gt=cv2.imread(
        gt_path,
        0
    )


    gt=cv2.resize(
        gt,
        (512,512),
        interpolation=cv2.INTER_NEAREST
    )


    gt=(
        gt>0
    ).astype(
        np.uint8
    )




    # =========================
    # Confusion Matrix
    # =========================


    TP += np.logical_and(
        pred==1,
        gt==1
    ).sum()



    FP += np.logical_and(
        pred==1,
        gt==0
    ).sum()



    FN += np.logical_and(
        pred==0,
        gt==1
    ).sum()



    TN += np.logical_and(
        pred==0,
        gt==0
    ).sum()




    # =========================
    # IoU
    # =========================


    intersection=np.logical_and(
        pred,
        gt
    ).sum()


    union=np.logical_or(
        pred,
        gt
    ).sum()



    iou=intersection/(union+1e-9)



    records.append(
        [
            name,
            iou
        ]
    )




    # =========================
    # Visualization
    # =========================


    vis=cv2.resize(
        img,
        (512,512)
    )


    color=np.zeros_like(
        vis
    )


    # BGR 蓝色

    color[pred==1]=(
        255,
        0,
        0
    )


    alpha=0.35



    overlay=cv2.addWeighted(
        vis,
        1-alpha,
        color,
        alpha,
        0
    )


    cv2.imwrite(
        os.path.join(
            SAVE_DIR,
            name
        ),
        overlay
    )




# =====================================================
# Final Metrics
# =====================================================


Precision=TP/(TP+FP+1e-9)


Recall=TP/(TP+FN+1e-9)



F1=(
    2*Precision*Recall/
    (Precision+Recall+1e-9)
)



IoU=(
    TP/
    (TP+FP+FN+1e-9)
)



IoU_bg=(
    TN/
    (TN+FP+FN+1e-9)
)



mIoU=(
    IoU+IoU_bg
)/2



Dice=(
    2*TP/
    (2*TP+FP+FN+1e-9)
)



Pixel_Accuracy=(
    TP+TN
) / (
    TP+TN+FP+FN+1e-9
)



FPS=(
    num_images/
    total_time
)




print("="*60)

print("SegFormer-B0 Pixel Evaluation")


print(
    f"Params(M)       : {params/1e6:.2f}"
)


print(
    f"FPS             : {FPS:.2f}"
)


print(
    f"Precision       : {Precision:.4f}"
)


print(
    f"Recall          : {Recall:.4f}"
)


print(
    f"F1-score        : {F1:.4f}"
)


print(
    f"IoU             : {IoU:.4f}"
)


print(
    f"mIoU            : {mIoU:.4f}"
)


print(
    f"Dice            : {Dice:.4f}"
)


print(
    f"Pixel Accuracy  : {Pixel_Accuracy:.4f}"
)


print("="*60)




# =====================================================
# CSV
# =====================================================


df=pd.DataFrame(
    records,
    columns=[
        "image",
        "IoU"
    ]
)



df.to_csv(
    CSV_PATH,
    index=False
)



print("CSV saved:")
print(
    os.path.abspath(CSV_PATH)
)


print("Visualization:")
print(
    os.path.abspath(SAVE_DIR)
)