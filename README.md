## awesome-yolo🎈🎈🎈

<div align=“center”>

🚀**yolo系列v3、v4、v5、v6、v7、x、r**以及其他改进的网络结构

#### awesome-yolo中支持的模块有：

- [x] 包含主流的 YOLOv3 模型网络结构中；
- [x] 包含主流的 YOLOv4 模型网络结构中；
- [x] 包含主流的 Scaled_YOLOv4 模型网络结构中；
- [x] 包含主流的 YOLOv5 模型网络结构中；
- [x] 包含主流的 YOLOv6 模型网络结构中；
- [x] 包含主流的 YOLOv7 模型网络结构中；
- [x] 包含主流的 YOLOX 模型网络结构中；
- [x] 包含主流的 YOLOR 模型网络结构中；
- [x] 包含transformer架构的backbone、neck、head；
- [x] 包含改进的transformer系列的backbone、neck、head；
- [x] 包含Attention系列的backbone、neck、head；
- [x] 包含基于anchor-free和anchor-based的检测器；  
...
...
> 更新中

## 🚀 可选择的组合
#### 🌟损失函数
* **CIoU（默认）**
```python
# 训练代码
python train.py --loss_category CIoU
```
* **DIoU**
```python
# 训练代码
python train.py --loss_category DIoU
```
* **GIoU**
```python
# 训练代码
python train.py --loss_category GIoU
```
* **EIoU**
```python
# 训练代码
python train.py --loss_category EIoU
```
* **SIoU**
```python
# 训练代码
python train.py --loss_category SIoU
```
______________________________________________________________________


#### 🔥YOLO系列热力图可视化
<div align=“center”>
<img src='https://github.com/Him-wen/OD_Heatmap/raw/main/images/bus.jpg' width="200px">
<img src='https://github.com/Him-wen/OD_Heatmap/raw/main/outputs/bus/1_bus-res.jpg' width="200px">
<div>

[🔗OD_Heatmap链接](https://github.com/Him-wen/OD_Heatmap)


## 🍋 网络模型架构图
* [yolov7](https://github.com/Him-wen/awesome-yolo#yolov7)
* [yolov6](https://github.com/Him-wen/awesome-yolo#yolov7)
* [yolox](https://github.com/Him-wen/awesome-yolo#yolox)
* [yolov5](https://github.com/Him-wen/awesome-yolo#yolov5)
* [yolor](https://github.com/Him-wen/awesome-yolo#yolor)
* [pp-yoloe](https://github.com/Him-wen/awesome-yolo#pp-yoloe)
* [pp-yolo2](https://github.com/Him-wen/awesome-yolo#pp-yolo2)
* [pp-yolo](https://github.com/Him-wen/awesome-yolo#pp-yolo)
* [scaled_yolov4](https://github.com/Him-wen/awesome-yolo#scaled_yolov4)
* [yolov4](https://github.com/Him-wen/awesome-yolo#yolov4)
* [yolov3](https://github.com/Him-wen/awesome-yolo#yolov3)  
更新...

### YOLOv7🚀🎈

<img src='docs/image/yolov7_model.jpg'>


______________________________________________________________________

### YOLOv6🚀🎈

<img src='docs/image/yolov6_model.jpg'>

______________________________________________________________________

### YOLOX🚀🎈

<img src='docs/image/yolox_model.png'>

______________________________________________________________________

### YOLOv5🚀🎈

<img src='docs/image/yolov5_model.jpg'>

______________________________________________________________________

### YOLOR🚀🎈

<img src='docs/image/yolor_model.jpg'>


______________________________________________________________________

### PP-YOLOE🚀🎈

<img src='docs/image/ppyoloe_model.png'>

______________________________________________________________________

### PP-YOLO2🚀🎈

<img src='docs/image/ppyolo2_model.png'>

______________________________________________________________________

### PP-YOLO🚀🎈

<img src='docs/image/ppyolo_model.png'>

______________________________________________________________________


### Scaled_YOLOv4🚀🎈

<img src='docs/image/scaled_yolov4.png'>


______________________________________________________________________

### YOLOv4🚀🎈

<img src='docs/image/yolov4_model.png'>

______________________________________________________________________

### YOLOv3🚀🎈

<img src='docs/image/yolov3_model.jpg'>

以上网络模型结构图来自以下参考链接🔗
[链接1](https://mp.weixin.qq.com/s/VEcUIaDrhc1ETIPr39l4rg)  [链接2](https://mp.weixin.qq.com/s/DFSROue8InARk-96I_Kptg)  [链接3](https://blog.csdn.net/qq_37541097/article/details/125132817)  [链接4](https://blog.csdn.net/qq_37541097/article/details/123594351)  [链接5](https://zhuanlan.zhihu.com/p/524548477)  [链接6](https://arxiv.org/abs/2011.08036)  [链接7](https://blog.csdn.net/qq_37541097/article/details/123229946)  [链接8](https://zhuanlan.zhihu.com/p/143747206)

______________________________________________________________________

## 🍉 Documentation
[model配置yaml文件](docs/model.md)
<details open>
<summary>教程</summary>

- [训练自定义数据](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)  🚀 推荐
- [获得最佳训练效果的技巧](https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results)  ☘️ 推荐
- [使用 Weights & Biases 记录实验](https://github.com/ultralytics/yolov5/issues/1289)  🌟 新
- [Roboflow：数据集、标签和主动学习](https://github.com/ultralytics/yolov5/issues/4975)  🌟 新
- [多GPU训练](https://github.com/ultralytics/yolov5/issues/475)
- [PyTorch Hub](https://github.com/ultralytics/yolov5/issues/36)  ⭐ 新
- [TFLite, ONNX, CoreML, TensorRT 导出](https://github.com/ultralytics/yolov5/issues/251) 🚀
- [测试时数据增强 (TTA)](https://github.com/ultralytics/yolov5/issues/303)
- [模型集成](https://github.com/ultralytics/yolov5/issues/318)
- [模型剪枝/稀疏性](https://github.com/ultralytics/yolov5/issues/304)
- [超参数进化](https://github.com/ultralytics/yolov5/issues/607)
- [带有冻结层的迁移学习](https://github.com/ultralytics/yolov5/issues/1314) ⭐ 新
- [YOLOv5架构概要](https://github.com/ultralytics/yolov5/issues/6998) ⭐ 新

</details>

______________________________________________________________________

## 🎓 Acknowledgement

<details><summary> <b>Expand</b> </summary>

* [AlexeyAB.darknet](https://github.com/AlexeyAB/darknet)
* [yolov3](https://github.com/ultralytics/yolov3)
* [yolov4](https://github.com/WongKinYiu/PyTorch_YOLOv4)
* [scaled_yolov4](https://github.com/WongKinYiu/ScaledYOLOv4)
* [yolov5](https://github.com/ultralytics/yolov5)
* [yolov6](https://github.com/meituan/YOLOv6)
* [yolov7](https://github.com/WongKinYiu/yolov7)
* [yolor](https://github.com/WongKinYiu/yolor)
* [yolox](https://github.com/Megvii-BaseDetection/YOLOX)
* [yolou](https://github.com/jizhishutong/YOLOU)
* [attention](https://github.com/xmu-xiaoma666/External-Attention-pytorch)
</details>

______________________________________________________________________

## 🌰 Statement
<details><summary> <b>Expand</b> </summary>

* The content of this site is only for sharing notes. If some content is infringing, please use issue to contact to delete it