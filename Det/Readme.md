# 本软件用于目标检测标注
# 采用TagImage.py进行目标标注
## 注意事项：
1. 标注前：
    1. 更改*IMG_PATH_DIR*为图像所在路径
    2. 更改*CLASSES*为所标注目标类别
    3. 更改*MODE*为所使用的图片读取库
    4. 更改*IMG_FORMAT*为所需要标注的图像格式 
    5. 更改*SCALARS*为各个类别标注框对应的颜色
2. 具体操作：
    1. 使用鼠标进行标注
    2. Q上一个文件夹，E下一个文件夹 
    3. d或D键调出下一标注图，并确认当前标注
    4. a或A键调出上一标注图，在当前图操作后按d或D键确认
    5. Z取消一个标注 
    6. C切换图片显示模式 
    7. F1, F2, ... 切换标注类别 
    8. ESC或关闭窗口键退出标注

GF6_WFV_E131.8_N31.4_20200918_L1A1120036753\GF6_WFV_E131.8_N31.4_20200918_L1A1120036753-2_10240_11264.tiff
## Version 4.2
1. 可以递归遍历任意文件夹

copyright by KeyanChen 20230223

## Version 3.0 
1. 添加图像宽高缩放因子缩放

copyright by KeyanChen 20191201

## Version 2.0 
1. 添加调出上一标注图和下一标注图
2. 使标注框规范化，无论从左上到右下还是其他方式都规范化到从左上到右下的标注方式  

copyright by KeyanChen 20190612

## Version 1.0 
1. 基本标注功能

copyright by KeyanChen 20190415 







