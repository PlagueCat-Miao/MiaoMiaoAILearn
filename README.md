# MyAILearn
近期AI调研喵

## 内容引导



## 调研

### AI 代码能干什么

| 名称                                                         | 功能                                                         | 开源情况       | 备注 |
| ------------------------------------------------------------ | ------------------------------------------------------------ | -------------- | ---- |
| [Tabnine](https://hub.tabnine.com/v9/home?tabnineUrl=http%3A%2F%2F127.0.0.1%3A1123%2Fsec-suvcvcabibtzwgascqxx) | 1. 全线代码完成(line)<br>2. 全功能代码完成(function)<br/>3. 自然语言到代码的补全<br/>4.学习你的编码模式和风格 | 闭源，免费     |      |
| [copilot](https://github.com/features/copilot/)              | 1. 创建样板和重复的代码模式,以一条注释来描述所想逻辑<br/>2.快速循环上下文代码行，提供多行的代码补全建议。 | 闭源，收费     |      |
| [阿里Cosy](https://github.com/alibaba-cloud-toolkit/cosy)    | 1. 行代码补全<br/>2.自然语言生成代码、及**文档（来自stackoverflow.com、阿里云开源社区、csdn）**应该是现场搜的<br/> | 闭源，免费     |      |
| 华为PanGu-Coder                                              | 1.能用中文！                                                 | 闭源，没见能用 |      |


### 用什么框架

| 框架                                                         | 开源协议       |                                                              |      | 备注 |
| ------------------------------------------------------------ | -------------- | ------------------------------------------------------------ | ---- | ---- |
| **[PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)**百度飞浆 | Apache-2.0[^1] |                                                              |      |      |
| PyTorch脸书                                                  |                |                                                              |      |      |
| TensorFlow谷歌                                               |                | [资料](https://github.com/aymericdamien/TensorFlow-Examples) |      |      |
|                                                              |                |                                                              |      |      |



### 用什么模型

- SOTA生成式模型
  - Text-to-Text：ChatGPT、LaMDA和PEER
  - Text-to-Code：Codex（GPT3）、Alphacode
  - 注释：RoBERTa（BERT）是**掩码语言建模（完形填空）**、GPT是**自回归语言建模（预测内容）**
- 开源模型网站：
  - [Hugging Face](https://huggingface.co/)我关注以下：
    - [codegen-350M-multi](https://huggingface.co/Salesforce/codegen-350M-multi)
    - [CodeGPT-small-java-adaptedGPT2](https://huggingface.co/microsoft/CodeGPT-small-java-adaptedGPT2/tree/main)
    - microsoft/CodeGPT-small-java
  


### 怎么训练

- 数据集如何整理、收集

### 理论支持

#### 1. 什么是 Transformers

Transformers是TensorFlow 2.0和PyTorch的最新自然语言处理库

Transformers(以前称为pytorch-transformers和pytorch-pretrained-bert)提供用于自然语言理解(NLU)和自然语言生成(NLG)的最先进的模型(BERT，GPT-2，RoBERTa，XLM，DistilBert，[XLNet](https://pytorchchina.com/tag/xlnet/)，CTRL …) ，拥有超过32种预训练模型，支持100多种语言，并且在TensorFlow 2.0和PyTorch之间具有深厚的互操作性。

#### 2.什么GTP

他怎么就使用了解码器


-----
[^1]: Apache License 2.0

