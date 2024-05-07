# Create environment name based on the exercise name
mamba create -y -n 03_learned_representations python=3.9
mamba activate 03_learned_representations

# Install requirements
mamba install -y numpy matplotlib pandas ipykernel ipympl tqdm scikit-learn scipy
mamba install -y pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
pip install umap-learn
# install the kernel for this env
python -m ipykernel install --user --name 03_learned_representations --display-name 03_learned_representations

# Return to base environment
mamba activate base
# we need ipympl to have interactive plots
mamba install -y jupyterlab ipympl