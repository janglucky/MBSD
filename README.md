# Official Implementation of MBSD
## ⚙️ Dependencies and Installation
```
# clone this respository
git clone xxx.git
cd xxx

# create environment with python=3.10
conda create -n brdm python=3.10
codna activate brdm
pip install -r requirements.txt
```

# 🚀 quick start
## Download the pretrained models
- Downoad the pretrained SD-2-base models from [HugginFace](https://huggingface.co/stabilityai/stable-diffusion-2-base)
- Downlaod the DAPE models from [Google Drive](https://drive.google.com/drive/folders/12HXrRGEXUAnmHRaf0bIn-S8XSK4Ku0JO).
- Download our pretrained models from [BRDM.zip](www.baidu.com) and unzip to root dir of BLANet

## Execute the test scripts
rememenber to modify the datasets path in `run.sh` to yours and execute the following instructions in command line.
```
cd /path/to/BLANet
bash run.sh
```