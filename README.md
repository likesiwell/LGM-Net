# LGM-Net
TensorFlow source code for the following publication:
> LGM-Net: Learning to Generate Matching Networks for Few-Shot Learning
>
> Huaiyu Li, Weiming Dong, Xing Mei, Chongyang Ma, Feiyue Huang, Bao-Gang Hu
>
> In *Proceedings of the 36th International Conference on Machine Learning (ICML 2019)*

# Requirements
- Python 3.5
- [NumPy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
- [tqdm](https://pypi.python.org/pypi/tqdm)
- [Tensorflow 1.4](https://www.tensorflow.org/install/)
- [Opencv 3.2.0](https://opencv.org/)

## Preparation
Set the path of resized miniImageNet dataset in `data.py`

## Train
```
python train_meta_matching_network.py --way 5 --shot 1

python train_meta_matching_network.py --way 5 --shot 5
```
## Test
```
python train_meta_matching_network.py --way 5 --shot 1 --is_test True --ckp checkpoint_id
```

## Acknowledgements
Thanks to [Antreas Antoniou](https://github.com/AntreasAntoniou/) for his [Matching Networks implementation](https://github.com/AntreasAntoniou/MatchingNetworks) of which parts were used for this implementation.
