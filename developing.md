git config --global user.name "Runpod"
git config --global user.email "runpod@chibifire.com"

dagster asset materialize -f  train_autoencoder.py --select autoencoder_512
