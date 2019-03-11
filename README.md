# 中文语音识别

## 模型简介

模型输入是一段不长于10秒钟的语音，模型的输出是该语音所对应的拼音标签。

模型参考了Baidu Deep Speech 2：http://proceedings.mlr.press/v48/amodei16.pdf

## 训练数据

所用的训练数据包含两个部分：

1. aishell-1语音数据集

AISHELL-ASR0009-OS1录音时长178小时，约14万条语音数据，下载地址：http://www.aishelltech.com/kysjcp

2. YouTube视频及对应字幕文件

从YouTube上获取MP4视频文件后转化成wav音频，同时使用对应的srt字幕文件作为target。总计时长大约120小时，有约20万条语音数据。如果有需要，请联系我的邮箱获取下载链接：chenmingxiang110@gmail.com
