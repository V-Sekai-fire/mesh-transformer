```bash
pip install nbconvert jupyterlab
jupyter nbconvert --execute mesh_train.ipynb
git config --global user.name "runpod"
git config --global user.email "runpod@chibifire.com"
ssh-keygen -t ed25519 -C "runpod@chibifire.com"
git clone git@spark.chibifire.com:ernest.lee/mesh-transformer-datasets.git
```