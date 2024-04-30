# Exercise: Transformer Representation 
#### An exercise for utilizing features extracted from a vision transformer for downstream tasks.

This exercise has two parts. In the first part, we'll learn how to extract features for a batch of images from the DINOv2 vision transformer model, and apply dimentionality reduction and clustering on those features.  
In the second part, we will train a model on top of those extracted features for the segmentation task.

### Setup
All the neccessary files are included in this repo. You just need to setup the python environment by running this script:
```bash
source setup.sh
```
After this, make sure you are in the `base` environment and then run `jupyter lab`:
```bash
mamba activate base
jupyter lab
```

### TA Info
To convert solutions python files into notebooks and generate the exercises, first, please install `jupytext` and `nbconvert`. Afterward, run `python ./generate_exercise <input_file.py>` .
