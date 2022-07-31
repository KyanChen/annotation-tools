# 本软件用于图像分割标注
# 采用AnnotateSegMask.py进行目标标注
## 注意事项：
1. 标注前：
    1. 更改*img_path_dir*为图像所在路径，可以递归遍历任意文件夹
    2. 更改*idx2color*为所标注目标类别编号及对应的伪彩色像素值
    3. [Optional]更改*ckpt*为训练的分割模型
2. 具体操作：
    1. 使用鼠标画框进行标注，使用Ctrl+P 打开自定义模式
    2. 调整参数，得到到满意的Box Mask后，按Y确定当前标注
    3. 按以上方式完成当前图片的所有Box-Level标注
    4. [Optional]如果误操作Y，可以按Z取消上一次Box-Level标注
    5. [Optional]如果需要删除部分区域标注结果，可以用鼠标画出Box框，然后按D删除该区域
    6. 按N切换下一张图
    7. 完成所有标注后，可以ESC退出

3. 操作方式：
   Ctrl+P 打开自定义模式
   N: Next 下一个文件 
   P: Previous下一个文件
   Z: 取消一个已有的标注
   D: 删除一个指定BOX标注
   Y: 确定一个标注
   S: 保存当前标注
   ESC或关闭窗口键退出标注
   
   

## Version 1.0 
1. 应对异常区域分割的基本标注功能实现

copyright by KyanChen 20220731