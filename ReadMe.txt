


By Linlin Wang (e-mail:wll_hit@163.com), Junping Zhang, Qingle Guo and Dong Chen

The Pytorch implementation for：IFTSDNet: An Interact-Feature Transformer Network with Spatial Detail Enhancement Module for Change Detection, published on IEEE Geoscience and Remote Sensing Letters.

[27 Dec. 2023] Release the code of IFTSDNet model

1. Dataset Ｄownload
LEVIR-CD：https://justchenhao.github.io/LEVIR/
DSIFN-CD：https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images/tree/master/dataset

Note: Please crop the LEVIR-CD dataset to a slice of 256×256 before training with it.

2. Dataset Path Setteing
LEVIR-CD or DSIFN-CD
     |—train
          |   |—A
          |   |—B
          |   |—OUT
     |—val
          |   |—A
          |   |—B
          |   |—OUT
     |—test
          |   |—A
          |   |—B
          |   |—OUT
Where A contains images of first temporal image, B contains images of second temporal images, and OUT contains groundtruth maps.

3. Traing and test Process

python train.py
python test.py

4. Revised parameters
You can revised related parameters in the "metedata.json" file.

5. Requirement

-Pytorch 1.8.0
-torchvision 0.9.0
-python 3.8
-opencv-python  4.5.3.56
-tensorboardx 2.4


6. Citation

If you use this code for your research, please cite our papers.

```
@Article{
AUTHOR = {Linlin Wang, Junping Zhang, Qingle Guo and Dong Chen},
TITLE = {IFTSDNet: An Interact-Feature Transformer Network with Spatial Detail Enhancement Module for Change Detection },
JOURNAL = {IEEE Geoscience and Remote Sensing Letters},
VOLUME = {20},
YEAR = {2023},
ISSN = {1-5},
}

```
7. Acknowledgments

Our code is inspired and revised by [pytorch-MSPSNet] and [pytorch-MSCANet], Thanks Qingle Guo and Mengxi Liu for their great work!!

8. Reference
[1] Q. Guo, et al., “Deep Multiscale Siamese Network with Parallel Convolutional Structure and Self-Attention for Change Detection,” IEEE Trans. Geosci. Remote Sens., vol. 60, pp. 1-12, 2022.
[2] M. Liu, et al., “A CNN-Transformer Network with Multiscale Context Aggregation for Fine-Grained Cropland Change Detection,” IEEE J. Sel. Topics Appl. Earth Observ. Remote Sens., vol. 15, pp. 4297-4306, 2022.
[3] S. Fang, K. Li, J. Shao and Z. Li, “SNUNet-CD: A Densely Connected Siamese Network for Change Detection of VHR Images,” IEEE Geosci. Remote Sens. Lett., vol. 19, pp. 1-5, Feb. 2021.