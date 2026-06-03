⭐ If this project helps your research, please consider giving it a star.
基础模型参考引用源码:https://github.com/bubbliiiing/unet-pytorch
## 引用
如果本工具对您的研究有所帮助，欢迎引用。
# 基于语义分割模型的半自动标注工具
## 项目简介
本项目基于已有语义分割模型（UNet及其改进模型）实现图像的半自动标注。
工具利用训练完成的分割模型对原始图像进行推理，自动生成符合 Labelme 格式的 JSON 标注文件，并通过轮廓提取、多边形简化和孔洞修复等策略生成可编辑标注结果。
生成的 JSON 文件可直接导入 Labelme 进行人工审核和修正，从而在保证标注质量的前提下显著降低人工标注工作量。
在实际大豆机收质量检测数据集构建过程中，本工具可减少约95%的人工标注工作量。图像预测实例在img文件夹中
---
## 主要特点
### 1. 无需额外开发环境
直接基于已有语义分割模型运行：
* UNet
* 其它输出分割掩膜的模型
无需额外部署标注平台。(如果你现在能够训练你自己的模型的话，那么环境应该就是满足要求的)
---
### 2. 自动生成 Labelme 标注文件
输出格式：
```text
xxx.jpg
xxx.json
```
生成结果可直接在 Labelme 中打开。
---
### 3. 智能轮廓提取
自动完成：
* 连通区域提取
* 孔洞填充
* 小目标过滤
* 多边形轮廓生成
提高标注完整性。
---
### 4. 自适应轮廓简化
根据目标面积和边界复杂度动态调整轮廓简化参数（Douglas-Peucker）。
优势：
* 保留关键边界特征
* 降低冗余顶点数量
* 减小 JSON 文件体积
---
### 5. 批量自动处理
支持：
* JPG
* PNG
批量推理并自动生成对应 JSON 文件。
---
## 工作流程
原始图像
↓
语义分割模型推理
↓
预测掩膜生成
↓
轮廓提取与优化
↓
JSON标注文件生成
↓
Labelme人工审核修正
↓
高质量数 据集
---
## 实际效果
在大豆机收质量检测任务中：
* 自动生成标注覆盖率 > 95%
* 人工修正时间降低约95%
* 最终标注质量可达到人工标注水平
该方法特别适用于：
* 农业表型分析
* 高光谱图像处理
* 目标分割数据集扩充
* 小目标标注任务
---
## 使用方法
### 1. 配置模型
```
下载权重：我用夸克网盘分享了「model_data.zip」，点击链接即可保存。打开「夸克APP」，无需下载在线播放视频，畅享原画5倍速，支持电视投屏。
链接：https://pan.quark.cn/s/e0840115081d
提取码：Fbc7
解压后直接放在unet文件夹中替换即可
将数据集图像放在"unet\VOCdevkit\VOC2007\JPEGImages"文件夹中
将掩码放在"unet\VOCdevkit\VOC2007\SegmentationClass"文件夹中，
运行文件voc_annotation划分数据集，目前的划分数据集比例为8：1：1
运行train.py训练后会自动保存权重在"log"文件夹中
至此，即可开始剩余未标注数据集的自动标注
python
model = Unet(
    model_path="logs/best_epoch_weights.pth",
    num_classes=4,
    input_shape=[512,512],
    backbone="vgg"
)
```
### 2. 配置路径
```python
image_folder = r"your_image_folder"
output_folder = r"your_output_folder"
```
### 3. 运行脚本
```bash
python jpgtojsonnew.py
```
生成结果：
```text
image1.jpg
image1.json

image2.jpg
image2.json
```
## 应用场景
* 农业图像标注
* 高光谱图像数据集构建
* 作物表型分析
* 病虫害识别
* 语义分割数据集扩充
* 主动学习与迭代标注
---
