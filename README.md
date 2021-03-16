# HandTailor

This repository is the implementation code and model of the paper "HandTailor: Towards High-Precision Monocular 3D Hand Recovery" ([arXiv](https://arxiv.org/abs/2102.09244))

<img src="demo.gif" width="480">

## Get the Code
```bash
git clone https://github.com/LyuJ1998/HandTailor.git
cd HandTailor
```

## Install Requirements
Please install the dependencies listed in `requirements.txt`.
```bash
pip install -r requirements.txt
```

## Download Pretrain Model
Download the pretrain model from [Google Drive](https://drive.google.com/file/d/1ZrGYm6SxfMHd6fqsEhkK2Dmyp57SFEYk/view?usp=sharing), and put the `model.pt` in `./checkpoints`

## Demo
To process the image provided in `./demo`, run
```bash
python demo.py
```
You can also put your data in the fold `./demo`, but remember to use the proper camera intrinsic like
```bash
python demo.py --fx=612.0206 --fy=612.2821 --cx=321.2842 --cy=235.8609
```
If camera information is unavailable, run
```bash
python demo_in_the_wild.py
```
We recommand you to utilize the camera intrinsic, which will improve the performance a lot.

## Realtime Demo
To reconstruct the hand from image captured with a webcam,run the following command. Also remember to use the proper camera intrinsic, the following command is for RealSense D435
```bash
python app.py --fx=612.0206 --fy=612.2821 --cx=321.2842 --cy=235.8609
```
When camera information is absence
```bash
python app_in_the_wild.py
```
## Citation
If you find this work helpful, please consider citing us
```
@article{lv2021handtailor,
  title={HandTailor: Towards High-Precision Monocular 3D Hand Recovery},
  author={Lv, Jun and Xu, Wenqiang and Yang, Lixin and Qian, Sucheng and Mao, Chongzhao and Lu, Cewu},
  journal={arXiv preprint arXiv:2102.09244},
  year={2021}
}
```