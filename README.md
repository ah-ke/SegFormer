<div align="center">
<h1>Urban Waterlogging Detection: A Challenging Benchmark and Large-Small Model Co-Adapter [ECCV2024]</h1>

Suqi Song<sup>1†</sup>, Chenxu Zhang<sup>1†</sup>, Peng Zhang<sup>1</sup>, Pengkun Li<sup>2</sup>, Fenglong Song<sup>3</sup>, Lei Zhang<sup>1*</sup>

<sup>1</sup>Chongqing University,
<sup>2</sup>Huawei Technologies Co., Ltd.,
<sup>3</sup>Huawei Noah's Ark Lab

<div>
<sup>†</sup> Equal contribution
<sup>*</sup> Corresponding author
</div>

<div>
  {songsuqi, zhangpeng}@stu.cqu.edu.cn, {zhangchenxu, leizhang}@cqu.edu.cn, {lipengkun3, songfenglong}@huawei.com
</div>

## Abstract

To address the problems of low efficiency of manual inspection and the difficulty of accurately obtaining waterlogging extents using traditional object detection methods in urban road waterlogging monitoring, a SegFormer-based intelligent identification method for urban road waterlogging is proposed. The SegFormer semantic segmentation network is employed to achieve pixel-level identification of waterlogged areas in complex road environments through multi-scale feature extraction and global context modeling. Experiments are conducted based on the training set of the public UW-Bench dataset and a self-constructed urban road waterlogging image test set, with comparisons against U-Net, Mask R-CNN, YOLOv8-seg, PSPNet, and DeepLabV3. The results demonstrate that SegFormer achieves superior performance in urban road waterlogging identification, with a precision of 94.50%, recall of 93.62%, F1-score of 94.06%, and mIoU of 93.08%. The model contains 3.71 M parameters and achieves an inference speed of 63.07 frames/s for a single 512×512 pixel image. The proposed method can effectively extract the spatial distribution characteristics of urban road waterlogging and provide technical support for urban flood monitoring, risk assessment, and smart water management while maintaining high identification accuracy and real-time performance.
<div align="center">
  <img src="pictures/fig1_bluemask_0307v2.jpg">
</div>

## Overview

* We propose an innovative large-small model co-adapter paradigm (LSM-adapter), aiming at achieving win-win regime. In order to learn a robust prompter, a Triple-S prompt adapter (TSP-Adapt) with a dynamic prompt combiner is formulated, enabling a success on adaptation. We pioneer the use of vision foundation model i.e., SAM for urban waterlogging detection, providing new insights for future research.

<div align="center">
  <img src="pictures/framework.jpg">
</div>
<p>
  The proposed Large-Small Model Co-adapter Paradigm, which include a histogram equalization adapter, 
  a triple-S prompt adapter and a dynamic prompt combiner. All components except the image encoder of 
  SAM are trained for prompt generation, learning and adaptation, toward adverse waterlogging detection.
</p>

* **Details of the proposed HE-Adapt and Semantic Prompter**

<div align="center">
  <img src="pictures/HE-SemP.jpg">
</div>
<p>
  The proposed histogram equalization adapter module mainly consists of a histogram equalization, a high-frequency filter and MLP blocks.
  Given that the features of water are not pronounced in most challenging scenarios, we first conduct histogram equalization operation to 
  highlight the contrast and texture of input image. %which can enhance the  of water, and make the boundaries more distinct. The enhanced 
  image is then passed through a high-frequency filter to extract high-frequency information beneficial for segmentation, and converted into 
  frequency patch embedding. The image embedding of large model contains rich semantic information. Therefore, we propose a prototype learning-based 
  semantic prompter, which leverages useful foreground features from large model to generate semantic prompts.
</p>

* **One-stage and Two-stage training strategies**

<div align="center">
  <img src="pictures/training.jpg">
</div>
<p>
  Two training strategies are proposed to explore suitable joint training of models with diverse architectures.
</p>

## UW-Bench Dataset
* Please note that</b> the training set ([Google Drive](https://docs.google.com/forms/d/e/1FAIpQLSfEP8b8D2MUJ23YbCmtrdc7-1-_8YH7bspRdwHYklpGR9L5zw/viewform?usp=dialog))



