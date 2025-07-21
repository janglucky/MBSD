# Official Implementation of MBSD
## ⚙️ Dependencies and Installation
```
# clone this respository
git clone git@github.com:janglucky/MBSD.git
cd MBSD

# create environment with python=3.10
conda create -n mbsd python=3.10
codna activate mbsd
pip install -r requirements.txt
```

# 🚀 quick start
## Download the pretrained models
- Download the pretrained SD-2-base models from [HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-2-base).
- Download the ram model from [HuggingFace](https://huggingface.co/spaces/xinyu1205/recognize-anything/blob/main/ram_swin_large_14m.pth).
- Download the DAPE model from [Google Drive](https://drive.google.com/drive/folders/12HXrRGEXUAnmHRaf0bIn-S8XSK4Ku0JO).
- Download our pretrained models from [Baidu Netdisk](https://pan.baidu.com/s/1xY8LPKk5C9jmbu26IVjK8w?pwd=68m8).

You can place the models in any folder, but don’t forget to update the path in run.sh.


## Execute the test scripts
Remember to update the dataset path in run.sh to your own and then run the following commands in the terminal.
```
cd /path/to/MBSD
bash run.sh
```