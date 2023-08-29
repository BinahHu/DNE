# DNE

This is the official implementation for "[Dense Network Expansion for Class Incremental Learning](https://arxiv.org/abs/2303.12696)" (CVPR2023)


## Environment setup

The dependencies are included in *requirements.txt*, which can be installed through:

```shell
pip install -r requirements.txt
```


## Dataset preparation

Please use the ```--data-path``` parameter in the bash file to config the root directory of your datasets.

[CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html) dataset will be automatically downloaded if it does not exist in the dataset root.

[ImageNet](https://www.image-net.org/) needs be manually downloaded to the dataset root.


## Training and evaluation

Please use ```train.sh``` and ```train_imagenet100.sh``` to train and evaluate the model on CIFAR100 and ImageNet100 datasets.

## Acknowledgement

This repo owes a huge thanks to the [dytox](https://github.com/arthurdouillard/dytox) repo.

## Citation

If this work is helpful to you, please  cite our work as:

```
@inproceedings{hu2023dense,
  title={Dense network expansion for class incremental learning},
  author={Hu, Zhiyuan and Li, Yunsheng and Lyu, Jiancheng and Gao, Dashan and Vasconcelos, Nuno},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11858--11867},
  year={2023}
}
```

## Contact

Feel free to contact ```z8hu@ucsd.edu``` if you have any question about this repo or our work.

## Future updates

We will continue polish this code repo to make it easier to read and reproduce.