# Developing

```bash
pip install nbconvert jupyterlab
jupyter nbconvert --execute mesh_train.ipynb
git config --global user.name "runpod"
git config --global user.email "runpod@chibifire.com"
ssh-keygen -t ed25519 -C "runpod@chibifire.com"
eval `ssh-agent`
ssh-add ~/.ssh/id_ed25519
git clone git@spark.chibifire.com:ernest.lee/mesh-transformer-datasets.git
```

## Python to ipython

```
pip install jupytext
jupytext --to py *.ipynb
rm -rf *.ipynb
```

## ipython to Python 

```
pip install jupytext
jupytext --to ipynb mesh_trainer.py
rm -rf *.py
```
