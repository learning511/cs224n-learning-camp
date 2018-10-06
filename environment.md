# DeepLearning-Environment
### Python

Python 能够使用各种各样的开发环境，这里我们强烈推荐使用 Anaconda 来进行Python 环境的管理，当然如果你有自己偏好的 Python 环境管理方式，你完全可以使用自己更喜欢的方式。

1.登录 Anaconda 的官网 [www.anaconda.com](www.anaconda.com)，选择下载

<img src="https://ws3.sinaimg.cn/large/006tNc79gy1fp8zfmxb5ej30lt0bb0tc.jpg" width='700'>

2.选择对应的操作系统

<img src="https://ws1.sinaimg.cn/large/006tNc79ly1fp8zgyn21kj31i60qr771.jpg" width="600">

3.选择 Python 3.6 的版本进行下载，因为 Python 2.7 不久之后很多开源库都不再继续支持，所以我们的整个课程都是基于 Python 3.6 开发的，请务必选择正确的 Python 版本，Python 3.6

<img src="https://ws2.sinaimg.cn/large/006tNc79ly1fp8zkhinhzj31i60r3400.jpg" width="600">

4.下载完成进行安装即可

### Jupyter 安装和环境配置

安装完成之后，liunx/mac 打开终端，windows打开 power shell，输入` jupyter notebook`就可以在浏览器打开交互的 notebook 环境，可以在里面运行代码

<img src="https://ws2.sinaimg.cn/large/006tNc79ly1fp8zmwu2fuj31880pogmh.jpg" width="700">



### CUDA

百度搜索 cuda，选择 CUDA Toolkit，进入 cuda 的官网，选择对应的操作系统进行下载

（注意 这里点进去直接是下载cuda9.1版本的，tensorflow 目前并不支持cuda9.1，我们可以从<https://developer.nvidia.com/cuda-toolkit-archive>中找到适合的cuda版本，例如cuda9.0等等。

<img src="https://ws2.sinaimg.cn/large/006tNc79gy1foalzdh3j2j31bi0ur0yh.jpg" width="500">

进入之后和后面即将介绍的安装过程相同）

看到下面可以进行的系统选择

<img src="https://ws4.sinaimg.cn/large/006tNc79gy1foalzkfgnjj31i60jg782.jpg" width='700'>

对于 cuda 的安装，不同的操作系统有着不同的安装方式，这里仅以 linux 环境举例（这是配置亚马逊云环境中的一部分），关于windows 的配置可以动手百度或者google，对于 mac 电脑，12 年之后就不再使用nvidia 的GPU，所以没有办法安装cuda。

建议使用云服务器或者安装 linux 双系统，可以省去很多麻烦，也有助于后期深度学习的开发。

 

选择 linux 对应的 cuda 下载 

<img src="https://ws2.sinaimg.cn/large/006tNc79gy1foam0d72vpj31cm1ms7q0.jpg" width="500">

在终端输入

```bash
$ wget https://developer.nvidia.com/compute/cuda/9.1/Prod/local_installers/cuda_9.1.85_387.26_linux
```

下载最新的 cuda 9，然后输入

```bash
$ bash cuda_9.1.85_387.26_linux
```

进行安装，接下来需要回答一些问题

```
accept/decline/quit: accept
Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 375.26?
(y)es/(n)o/(q)uit: y
Do you want to install the OpenGL libraries?
(y)es/(n)o/(q)uit [ default is yes ]: y
Do you want to run nvidia-xconfig?
(y)es/(n)o/(q)uit [ default is no ]: n
Install the CUDA 8.0 Toolkit?
(y)es/(n)o/(q)uit: y
Enter Toolkit Location
 [ default is /usr/local/cuda-8.0 ]:
Do you want to install a symbolic link at
/usr/local/cuda?
(y)es/(n)o/(q)uit: y
Install the CUDA 8.0 Samples?
(y)es/(n)o/(q)uit: n
```

运行完成之后就安装成功了，可以在终端输入

```bash
nvidia-smi
```

查看GPU，最后我们需要将 cuda 添加在系统环境变量中方便以后的安装中找到

```bash
echo "export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:/usr/local/cuda-9.1/lib64" >>~/.bashrc
source ~/.bashrc
```



### 深度学习框架 TensorFlow 和 PyTorch 安装

#### TensorFlow 安装

目前 Tensorflow 支持在 Linux, MacOS, Windows 系统下安装，有仅支持 CPU 的版本，在缺少 GPU 资源时是一个不错的选择，也有 GPU 版本的实现高性能 GPU 加速。

在安装 GPU 版本之前需要一些额外的环境

#### libcupti-dev

一行命令即可

```bash
$ sudo apt-get install libcupti-dev
```

#### cudnn

进入 https://developer.nvidia.com/cudnn，点击下载

<img src="https://ws4.sinaimg.cn/large/006tNc79gy1foam0z0fykj319y0hxn7a.jpg" width='500'>

会要求进行注册，点击 Join

<img src="https://ws2.sinaimg.cn/large/006tNc79gy1foam18kpadj30ic09faa9.jpg" width="250">



然后填写关于你的一些信息就完成了注册。然后就可以打开 Download 出现下面的页面并选择下载压缩包

<img src="https://ws2.sinaimg.cn/large/006tNc79gy1foam2cvftyj30qi0r5qaq.jpg" width="400">

解压后在当前目录运行下面命令即完成

```bash
$ sudo cp cuda/include/cudnn.h /usr/local/cuda/include 
$ sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64 
$ sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
```

#### 安装 Tensorflow

到这里 Tensorflow 的安装就非常简单了，可以在系统中用 pip 也可以在 anaconda 虚拟环境中安装

- pip 安装

  ```bash
  # 仅安装cpu版本 python2.x
  $ pip install tensorflow
  # python3.x
  $ pip3 install tensorflow
  # 安装gpu版本 python2.x
  $ pip install tensorflow-gpu
  # python3.x
  $ pip3 install tensorflow-gpu
  ```

- anaconda安装

  ```bash
  # 激活环境
  # 下面的`$YOUR_ENV`替换成你自己的，没有的话要生成一个新的环境，可以参考下面注释的例子
  # `conda create -n tensorflow pip python=2.7 # or python=3.3, etc.`
  # 这样会构建一个名为 tensorflow，python 是2.7版本的虚拟环境
  # 换名字很简单，换python版本的话也只需要将2.7改变即可，比如改变成3.6
  $ source activate $YOUR_ENV
  # 在环境中安装tensorflow，注意这里的tfBinaryURL需要根据需求替换，后面详述
  ($YOUR_ENV)$ pip install --ignore-installed --upgrade tfBinaryURL
  ```

  tfBinaryURL 以在https://tensorflow.google.cn/install/install_linux#the_url_of_the_tensorflow_python_package选择

#### 验证安装

终端中打开 python 解释器，运行下面命令成功即可

```python
# Python
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
```

#### 出现问题

- 更全面的 Tensorflow 安装页面 https://tensorflow.google.cn/install/
- 检查硬件配置是否满足需求，GPU版本的 Tensorflow 需要计算能力在 3.5 及以上的显卡，可以在这里 https://developer.nvidia.com/cuda-gpus 查到自己的显卡计算能力
- 在 Tensorflow 的 Github issues 里面寻找类似问题及解决方案

#### PyTorch 安装

目前 PyTorch 官方只支持linux 和 MacOS，如果要查看 windows 的安装方法，请看后面。

在 linux 和 MacOS 这两个系统下进行安装非常的简单，访问到官网

[www.pytorch.org](http://www.pytorch.org)

<img src="https://ws2.sinaimg.cn/large/006tNc79gy1foam3kqe43j31i60ncn3m.jpg" width="700">

按照提示在终端输入命令行即可 

#### 如何在 windows 下装 PyTorch
使用 windows 的同学可以访问这个链接查看如何在 windows 下面安装pytorch

https://zhuanlan.zhihu.com/p/26871672

 

#### 验证安装

终端中打开 python 解释器，运行下面命令成功即可

```python
# Python
import torch
x = torch.Tensor([3])
print(x)
# x_gpu = torch.Tensor([3]).cuda() # GPU 安装验证
# print(x)
```

