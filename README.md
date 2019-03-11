# 中文语音识别

## 模型简介

模型输入是一段不长于10秒钟的语音，模型的输出是该语音所对应的拼音标签。

模型参考了Baidu Deep Speech 2：http://proceedings.mlr.press/v48/amodei16.pdf

使用了CNN+GRU+CTC_loss的结构

## 训练数据

所用的训练数据包含两个部分：

1. aishell-1语音数据集

AISHELL-ASR0009-OS1录音时长178小时，约14万条语音数据，下载地址：http://www.aishelltech.com/kysjcp

2. YouTube视频及对应字幕文件

从YouTube上获取MP4视频文件后转化成wav音频，同时使用对应的srt字幕文件作为target。总计时长大约120小时，有约20万条语音数据。如果有需要，请联系我的邮箱获取下载链接：chenmingxiang110@gmail.com

## 使用方法

### 1. 训练模型

根据实际需求和硬件情况，可以选择需要的模型进行训练和调试。各个模型区别如下。如果实在含GPU的机器上训练模型，直接运行 train901.py，train902.py，或者train903.py 即可。如果是在CPU上训练，则运行 train901_cpu.py，train902_cpu.py，或者train903_cpu.py。

|模型名称 |CNN层数 |GRU层数 |GRU维度 |训练时间 |
|--- |--- |--- |--- |--- |
|901|2|3|256 |约30小时收敛|
|902|2|5|256 |约55小时收敛|
|903|2|5|1024|约130小时收敛|

这里的训练时间仅仅是一个大概的统计，训练使用一块Tesla V100完成。

### 2. 识别音频

## 效果和demo
