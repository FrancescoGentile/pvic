##
##
##

pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install torchmetrics==0.10.0 matplotlib==3.6.3 scipy==1.10.0 tqdm==4.64.1 numpy==1.24.1 timm==0.6.12 wandb==0.13.9 seaborn==0.13.0
git submodule init
git submodule update
pip install -e pocket