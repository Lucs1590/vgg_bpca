# Exploring BPCAPooling Layer in VGG-16 Network: A Comparative Study with Conventional Pooling Methods
This repository was created for the paper submitted to SIBGRAPI and contains the code for the experiments.
Our goal is to provide a simple and easy to use implementation of the proposed method.

The proposed method is called BPCAPooling, that is inspired by the conventional PCA method but specifically designed for pooling in CNNs. It subdivides the input feature map into blocks and applies PCA to each block to extract important information while preserving spatial information. The transformed blocks are then concatenated to form a reduced-size feature map. This is ilustrated in the figure below.

![BPCAPooling](https://raw.githubusercontent.com/Lucs1590/vgg_bpca/2c97a54297698363aa6349a655371a6eb3bbdb54/images/bpca.png?token=GHSAT0AAAAAAB5UG3TH7YCPUFGK7NJR3IL6ZE7LHZA)

## Installation
The code was tested with Python 3.6.9 and Tensorflow 2.12.0.
To install the required packages run:
```
pip install -r requirements.txt
```

## Usage
The code is structured based on the Jupiter notebooks in the `notebooks` folder. So the main files are the notebooks.

The notebooks contain the code for the experiments (using Food-101 and CIFAR-100) and can be run directly.

The `scripts` folder contains the implementation of the proposed method and other methods used in the experiments.

In the `models_weights` folder you could find the weights of the models used in the experiments on `h5` format, so you can load them directly in the notebooks.
**OBS.: In this repository we are GIT LFS to store the weights files. So you need to install it to download the weights.**

## Citation
If you use this code in your research, please cite our paper:
```
@inproceedings{,
  title={Exploring BPCAPooling Layer in VGG-16 Network: A Comparative Study with Conventional Pooling Methods},
  author={Lucas de Brito Silva, Alvaro Leandro Cavalcante Carneiro, Uemerson Pinheiro Junior, Denis Henrique Pinheiro Salvadeo, Davi Duarte de Paula},
  booktitle={},
  year={2023},
  organization={Sao Paulo State University (UNESP)},
  url={}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/Lucs1590/vgg_bpca/blob/master/LICENSE.md) file for details.
