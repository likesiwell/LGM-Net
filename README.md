# LGM-Net
Tensorflow code for ICML 2019 paper: LGM-Net: Learning to Generate Matching Networks for Few-Shot Learning


# Requirements
- Python 3.5
- [NumPy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
- [tqdm](https://pypi.python.org/pypi/tqdm)
- [Tensorflow 1.4](https://www.tensorflow.org/install/)
- [Opencv 3.2.0]

## Preparation
Set right resized $\textit{mini}ImageNet$ directory in data.py

## Train
python train_meta_matching_network.py --way 5 --shot 1

python train_meta_matching_network.py --way 5 --shot 5

## Test
python train_meta_matching_network.py --way 5 --shot 1 --is_test True --ckp 110

## Acknowledgements
Thanks to https://github.com/AntreasAntoniou/ for his Matching Networks implementation of which parts were used for this implementation. More details at https://github.com/AntreasAntoniou/MatchingNetworks

