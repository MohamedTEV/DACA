# Detect, Augment, Compose, and Adapt: Four Steps for Unsupervised Domain Adaptation in Object Detection

<p align="center">
  <img style="width: 100%" src="pipeline/teaser.png">
</p>
<br>

> **Detect, Augment, Compose, and Adapt: Four Steps for Unsupervised Domain Adaptation in Object Detection**<br>
> Mohamed Lamine Mekhalfi, Davide Boscaini, Fabio Poiesi <br>
> **BMVC 2023**

> Paper: [arXiv](https://arxiv.org/abs/2308.15353) <br>
> [Audio summary](https://sciencecast.org/casts/fkna59wuhtgl?t=DG2uvglJKzf29DxHzc3fZBP7%2BiQr8fR7x6FPu3sPIC8ytbpnNFEn9Vp68Z8Cb3vRnI4immjbS9QFlYtTx4uyyA%3D%3D)

> **Abstract:** *Unsupervised domain adaptation (UDA) plays a crucial role in object detection when adapting a source-trained detector to a target domain without annotated data. In this paper, we propose a novel and effective four-step UDA approach that leverages self- supervision and trains source and target data concurrently. We harness self-supervised learning to mitigate the lack of ground truth in the target domain. Our method consists of the following steps: (1) identify the region with the highest-confidence set of detections in each target image, which serve as our pseudo-labels; (2) crop the identified region and generate a collection of its augmented versions; (3) combine these latter into a com- posite image; (4) adapt the network to the target domain using the composed image. Through extensive experiments under cross-camera, cross-weather, and synthetic-to-real scenarios, our approach achieves state-of-the-art performance, improving upon the near- est competitor by more than 2% in terms of mean Average Precision (mAP). The source code will be made publicly available upon publication.*

# Setup
Clone repo:
```bash
git clone https://github.com/MohamedTEV/DACA.git
```
Navigate to the main directory:
```bash
cd DACA
```

Create environment and install dependencies:
```bash
conda create --name daca python=3.7
conda activate daca
pip install -r requirements.txt  
```

Install pytorch & torchvision (for the right version please visit: https://pytorch.org/):
```bash
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```

# Datasets download & preparation:
The datasets can be downloaded at the following sources. Please note that YOLO format is used for all the datasets.
  - Cityscapes and Foggy Cityscapes datasets can be downloaded at: [Cityscapes/Foggy](https://www.cityscapes-dataset.com/downloads/). 
  - Sim10k dataset can be downloaded at: [Sim10k](https://fcav.engin.umich.edu/projects/driving-in-the-matrix).
  - KITTI dataset can be downloaded at: [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d).
The images should be placed inside dataset/images/train (e.g., cityscapes/images/train) and the labels should be placed inside dataset/labels/train (e.g., cityscapes/labels/train). If the images are saved in a different destination, change the .yaml files inside ./data accordingly. 


# Training
Training DACA undergoes two steps, namely (i) a pre-adaptation in which the detector is trained on the source dataset only, and (ii) adaptation in which the model departs from the weights of previous step and performs adaptation. Oracle score is regarded as an 'upper bound' performance where the model is trained on the target dataset using groundtruth. For the pretrained YOLO weights, please visit [YOLOv5](https://github.com/ultralytics/yolov5/releases). Each of the previous operations, besides the validation step, can be executed by running the following bash scripts:
## pre-adaptation
```bash
Cityscapes(source):
python train.py --name cityscapes --epochs 20 --data data/cityscapes2foggy.yaml --weights yolov5s.pt
```

```bash
Sim10(source):  (only 'Car' class)
python train.py --name sim10k --epochs 20 --data data/sim10k2cityscapes.yaml --weights yolov5s.pt
```

```bash
KITTI(source):  (only 'Car' class)
python train.py --name kitti --epochs 20 --data data/kitti2cityscapes.yaml --weights yolov5s.pt
```


## DACA-based adaptation:
Cityscapes(source) -> Foggy Cityscapes(target):
```bash
python uda_daca_train.py --name cityscapes2foggy_daca_All --epochs 50 --data data/cityscapes2foggy.yaml --weights runs/train/cityscapes/weights/last.pt
```

Sim10(source) -> Cityscapes(target):  (only 'Car' class)
```bash
python uda_daca_train.py --name sim10k2cityscapes_daca_All --epochs 50 --data data/sim10k2cityscapes.yaml --weights runs/train/sim10k/weights/last.pt
```

KITTI(source) -> Cityscapes(target):  (only 'Car' class)
```bash
python uda_daca_train.py --name kitti2cityscapes_daca_All --epochs 50 --data data/kitti2cityscapes.yaml --weights runs/train/kitti/weights/last.pt
```

## Oracle:
```bash
python train.py --name oraclecityscapes --epochs 20 --data data/cityscapes.yaml --weights yolov5s.pt
```

```bash
python train.py --name oraclefoggy --epochs 20 --data data/foggycityscapes.yaml --weights yolov5s.pt
```


# Evaluation (DACA adaptation)
The following bash commands evaluate the DACA-adapted models, to evaluation pre-adapted models, change the weights path accordingly:

Cityscapes(source) -> Foggy Cityscapes(target):
```bash
python val.py  --name exp  --data data/cityscapes2foggy.yaml  --weights runs/train/cityscapes2foggy/weights/last.pt
```

Sim10(source) -> Cityscapes(target):  (only 'Car' class)
```bash
python val.py  --name exp  --data data/sim10k2cityscapes.yaml  --weights runs/train/sim10k2cityscapes/weights/last.pt
```

KITTI(source) -> Cityscapes(target):  (only 'Car' class)
```bash
python val.py  --name exp  --data data/kitti2cityscapes.yaml  --weights runs/train/kitti2cityscapes/weights/last.pt

```


# Citation

```
@inproceedings{mekhalfi2023daca,
  title={{Detect, Augment, Compose, and Adapt: Four Steps for Unsupervised Domain Adaptation in Object Detection}},
  author={Mekhalfi, Mohamed Lamine and Boscaini, Davide and Poiesi, Fabio},
  booktitle={BMVC},
  year={2023}
}

```

