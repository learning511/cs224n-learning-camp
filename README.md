# cs224n learning camp 

## 课程资料
1. [课程主页](https://web.stanford.edu/class/cs224n/)  
2. [中文笔记](http://www.hankcs.com/nlp/cs224n-introduction-to-nlp-and-deep-learning.html)  
3. [课程视频](https://www.bilibili.com/video/av9595652/)  
4. 环境配置（推荐使用Linux或者Mac系统，以下环境搭建方法皆适用）  
 - [本地环境](https://github.com/learning511/cs224n-learning-camp/blob/master/environment.md)
 - [Docker环境](https://github.com/ufoym/deepo)
6. [作业链接](https://github.com/learning511/cs224n-learning-camp/blob/master/Assignmnet.md) 


#### 重要🔥🔥一些的资源：
1. [深度学习斯坦福教程](http://deeplearning.stanford.edu/wiki/index.php/UFLDL%E6%95%99%E7%A8%8B)
2. [廖雪峰python3教程](https://www.liaoxuefeng.com/article/001432619295115c918a094d8954bd493037b03d27bf9a9000)
3. [github教程](https://www.liaoxuefeng.com/wiki/0013739516305929606dd18361248578c67b8067c8c017b000)
4. [莫烦机器学习教程](https://morvanzhou.github.io/tutorials)
5. [深度学习经典论文](https://github.com/floodsung/Deep-Learning-Papers-Reading-Roadmap.git)
6. [斯坦福cs229代码(机器学习算法python徒手实现)](https://github.com/nsoojin/coursera-ml-py.git)  
7. [本人博客](https://blog.csdn.net/dukuku5038/article/details/82253966)


## 前言
自然语言是人类智慧的结晶，自然语言处理是人工智能中最为困难的问题之一，而对自然语言处理的研究也是充满魅力和挑战的。
通过经典的斯坦福cs224n教程，让我们一起和自然语言处理共舞！也希望大家能够在NLP领域有所成就！

## 学员要求
- 具有python编程基础
- 了解并掌握高等数学、概率论、线性代数知识
- 了解并掌握基础机器学习算法：梯度下降、线性回归、逻辑回归、Softmax、SVM、PAC（先修课程斯坦福cs229 或者周志华西瓜书）
- 具有英语4级水平（深度学习相关学习材料、论文基本都是英文，**一定要阅读英文原文，进步和提高的速度会加快**）

## 知识复习
为了让大家逐渐适应英文阅读，复习材料我们有中英两个版本，**但是推荐大家读英文**
### 数学复习（斯坦福资料）
1.[线性代数](http://web.stanford.edu/class/cs224n/readings/cs229-linalg.pdf)  
2.[概率论](http://web.stanford.edu/class/cs224n/readings/cs229-prob.pdf)  
3.[凸函数优化](http://web.stanford.edu/class/cs224n/readings/cs229-cvxopt.pdf)  
4.[随机梯度下降算法](http://cs231n.github.io/optimization-1/)  

### Python复习（斯坦福资料）
[Python复习](http://web.stanford.edu/class/cs224n/lectures/python-review.pdf)

## 学习安排
### 阶段 1
1. 自然语言处理和深度学习简介  
- **课件:** [lecture01-introduction]()
- [观看视频](https://www.bilibili.com/video/av30326868/?spm_id_from=333.788.videocard.0)
- 学习笔记：[自然语言处理与深度学习简介](http://www.hankcs.com/nlp/cs224n-introduction-to-nlp-and-deep-learning.html)

2. 词的向量表示1：
- **课件:** [lecture02]() 
- [观看视频](https://www.bilibili.com/video/av30326868/?p=2)
- 学习笔记：[wordvecotor](http://www.hankcs.com/nlp/word-vector-representations-word2vec.html)

3. 论文导读：一个简单但很难超越的Sentence Embedding基线方法
- **课件:** [paper1]() 
- **论文笔记**：[Sentence Embedding](http://www.hankcs.com/nlp/cs224n-sentence-embeddings.html)
4. 作业：Assignment 1.1-1.2  
- 1.1 Softmax  
- 1.2 Neural Network Basics  
### 阶段 2
1.  高级词向量表示：word2vec 2
- **课件:** [lecture03]()
- [观看视频](https://www.bilibili.com/video/av30326868/?p=3)
- 学习笔记： [word2vec-2](http://www.hankcs.com/nlp/cs224n-advanced-word-vector-representations.html)

2. Word Window分类与神经网络
- **课件:** [lecture04]() 
- [观看视频](https://www.bilibili.com/video/av30326868/?p=4)
- 学习笔记：[Word Window分类与神经网络](http://www.hankcs.com/nlp/cs224n-word-window-classification-and-neural-networks.html)

3. 论文导读：词语义项的线性代数结构与词义消歧
- **课件:** [paper2](http://web.stanford.edu/class/cs224n/lectures/lecture2.pdf)  
-  **论文笔记：**[词语义项的线性代数结构与词义消歧](http://www.hankcs.com/nlp/cs224n-word-senses.html)
4. 作业：Assignment 1.3-1.4 
- 1.3 word2vec  
- 1.4 Sentiment Analysis  
### 阶段 3
1.  反向传播与项目指导：Backpropagation and Project Advice
- **课件:** lecture05
- [观看视频](https://www.bilibili.com/video/av30326868/?p=5)
- 学习笔记： [反向传播与项目指导](http://www.hankcs.com/nlp/cs224n-backpropagation-and-project-advice.html)

2. TensorFlow入门
- **课件:** lecture06
- [观看视频](https://www.bilibili.com/video/av30326868/?p=6)
- 学习笔记：[TensorFlow](http://www.hankcs.com/nlp/cs224n-tensorflow.html)

3. 论文导读：高效文本分类
- **课件:** [paper]
-  **论文笔记：**[高效文本分类](http://www.hankcs.com/nlp/cs224n-bag-of-tricks-for-efficient-text-classification.html)
4. 作业：Assignment 2.1  
- 2.1 Tensorflow Softmax

### 阶段 4
1.  反向传播与项目指导：Dependency Parsing 、、、、、、、、、、、、、、、
- **课件:** lecture07
- [观看视频](https://www.bilibili.com/video/av30326868/?p=7)
- 学习笔记： [句法分析和依赖解析](http://www.hankcs.com/nlp/cs224n-dependency-parsing.html)

2. RNN和语言模型
- **课件:** lecture08
- [观看视频](https://www.bilibili.com/video/av30326868/?p=8)
- 学习笔记：[RNN和语言模型](http://www.hankcs.com/nlp/cs224n-rnn-and-language-models.html)

3. 论文导读：词嵌入对传统方法的启发
- **课件:** [paper]
-  **论文笔记：**[词嵌入对传统方法的启发](http://www.hankcs.com/nlp/cs224n-improve-word-embeddings.html)
4. 作业：Assignment 2.2
- 2.2 Neural Transition-Based Dependency Parsing
### 阶段 5
1.  高级LSTM及GRU：LSTM and GRU
- **课件:** lecture09
- [观看视频](https://www.bilibili.com/video/av30326868/?p=9)
- 学习笔记： [高级LSTM及GRU](http://www.hankcs.com/nlp/cs224n-mt-lstm-gru.html)

2. 期中复习
- **课件:** lecture10
- [观看视频](https://www.bilibili.com/video/av30326868/?p=2)

3. 论文导读：基于转移的神经网络句法分析的结构化训练
- **课件:** [paper]
-  **论文笔记：**[基于转移的神经网络句法分析的结构化训练](http://www.hankcs.com/nlp/cs224n-syntaxnet.html)
4. 作业：Assignment 2.3
- 2.3 Recurrent Neural Networks: Language Modeling

### 阶段 6
1.  机器翻译、序列到序列、注意力模型：Machine Translation, Seq2Seq and Attention 
- **课件:** lecture11
- [观看视频](https://www.bilibili.com/video/av30326868/?p=11)
- 学习笔记： [机器翻译、序列到序列、注意力模型]http://www.hankcs.com/nlp/cs224n-9-nmt-models-with-attention.html)

2. GRU和NMT的进阶
- **课件:** lecture12
- [观看视频](https://www.bilibili.com/video/av30326868/?p=12)
- 学习笔记：[GRU和NMT的进阶](http://www.hankcs.com/nlp/cs224n-gru-nmt.html)

3. 论文导读：谷歌的多语种神经网络翻译系统
- **课件:** [paper]
-  **论文笔记：**[谷歌的多语种神经网络翻译系统](http://www.hankcs.com/nlp/cs224n-google-nmt.html)
4. 作业：Assignment 3.1
- 3.1  A window into NER

### 阶段 7
1.  语音识别的end-to-end模型
- **课件:** lecture13
- [观看视频](https://www.bilibili.com/video/av30326868/?p=13)
- 学习笔记： [语音识别的end-to-end模型](http://www.hankcs.com/nlp/cs224n-end-to-end-asr.html)

2. 卷积神经网络:CNN
- **课件:** lecture14
- [观看视频](https://www.bilibili.com/video/av30326868/?p=14)
- 学习笔记：[卷积神经网络](http://www.hankcs.com/nlp/cs224n-convolutional-neural-networks.html)

3. 论文导读：谷歌的多语种神经网络翻译系统
- **课件:** [paper]
-  **论文笔记：**[读唇术](http://www.hankcs.com/nlp/cs224n-lip-reading.html)
4. 作业：Assignment 3.2
- 3.2  Recurrent neural nets for NER


### 阶段 8
1.  Tree RNN与短语句法分析
- **课件:** lecture15
- [观看视频](https://www.bilibili.com/video/av30326868/?p=15)
- 学习笔记： [Tree RNN与短语句法分析](http://www.hankcs.com/nlp/cs224n-tree-recursive-neural-networks-and-constituency-parsing.html)

2. 指代消解
- **课件:** lecture16
- [观看视频](https://www.bilibili.com/video/av30326868/?p=16)
- 学习笔记：[指代消解](http://www.hankcs.com/nlp/cs224n-coreference-resolution.html)

3. 论文导读：谷歌的多语种神经网络翻译系统
- **课件:** [paper]
-  **论文笔记：**[Character-Aware神经网络语言模型](http://www.hankcs.com/nlp/cs224n-character-aware-neural-language-models.html)
4. 作业：Assignment 3.3
- 3.3  Grooving with GRUs


### 阶段 9
1.   DMN与问答系统
- **课件:** lecture17
- [观看视频](https://www.bilibili.com/video/av30326868/?p=17)
- 学习笔记： [DMN与问答系统](http://www.hankcs.com/nlp/cs224n-dmn-question-answering.html)

2.  NLP存在的问题与未来的架构
- **课件:** lecture18
- [观看视频](https://www.bilibili.com/video/av30326868/?p=18)
- 学习笔记：[指代消解](http://www.hankcs.com/nlp/cs224n-nlp-issues-architectures.html)

3. 论文导读：神经网络自动代码摘要
- **课件:** [paper]
-  **论文笔记：**[神经网络自动代码摘要](http://www.hankcs.com/nlp/cs224n-summarizing-source-code.html)
6. 比赛

### 阶段 10 前沿论文
1. 论文导读：neural-turing-machines
- **课件:** [paper]
-  **论文笔记：**[neural-turing-machines](http://www.hankcs.com/nlp/cs224n-neural-turing-machines.html)
2 论文导读： 深度强化学习用于对话生成
- **课件:** [paper]
-  **论文笔记：**[深度强化学习用于对话生成](http://www.hankcs.com/nlp/cs224n-deep-reinforcement-learning-for-dialogue-generation.html)
3. 图像对话：
- **课件:** [paper]
-  **论文笔记：**[图像对话](http://www.hankcs.com/nlp/cs224n-visual-dialog.html)
 ### 阶段 12
    比赛
