# %% [markdown]
# <h1 style="font-family: serif">Exercise: Transformer Representation</h1>

# %% [markdown]
# In this exercise, we'll learn how to extract features out of a pre-trained transformer model and use those features for downstream tasks. For this exercise, we'll use [*DINOv2: A Self-supervised Vision Transformer Model*](https://dinov2.metademolab.com/) by *meta*. This model is trained in a teacher-student paradigm, without any supervision, and it produces features suitable for different downstream tasks like image classification, depth estimation, semantic segmentation, etc.
# <br><br>**Note:** DINOv2 makes 14x14 patches out of an input image, and then produce features for each patch (not for each pixel).

# %% [markdown]
# <div class="alert alert-danger">
#     Please switch to the <code>05_learned_representations</code> environment.
# </div>

# %% [markdown]
# <h2 style="font-family: serif">I. Part One</h2>

# %% [markdown]
# In the first part, we will examine and visualize the extracted features using **PCA** and **UMAP**. At the end of this part, we use **KMeans** on top of the extracted features to cluster them, and to compare obtained clusters with given ground truth masks.

# %%
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision.transforms import v2 as tv_transforms2

import utils


# %%
# to have interactive plots
%matplotlib widget
plt.ioff()

SEED = 2024
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


# %%
# color map for visualization of the ground truth masks
cm, colors = utils.get_colormap()
cm

# %% [markdown]
# ## Data
# We are using data provided by [*Dense cellular segmentation for EM using 2D-3D neural network ensembles*](https://leapmanlab.github.io/dense-cell/).
# <br>The data contains *tiff* files in train and evaluation sets along with their ground truth masks. Masks include dense annotations for seven classes.
# <br>Images have a resolution of 800 x 800 pixels which are a bit large to fit in memory or GPU 😁 . However, we'll transform them into a smaller manageable resolution.

# %%
# the data resides in this path: "/group/dl4miacourse/platelet_data"
# load data and ground truth masks
data_images = utils.get_images_from_tiff(
    "/group/dl4miacourse/platelet_data/train-images.tif", to_rgb=True
)
gt_masks = utils.get_images_from_tiff(
    "/group/dl4miacourse/platelet_data/train-labels.tif", to_rgb=False
)

print(data_images.shape, gt_masks.shape)
utils.plot_data_sample(data_images[0], gt_masks[0], cmap=cm)

# %%
# original image size
image_size = data_images.shape[1]
# ground truth classes and their labels
num_classes = 7
classes = {
    "background": 0,
    "cell": 1,
    "mitochondrion": 2,
    "alpha granule": 3,
    "canalicular channel": 4,
    "dense granule": 5,
    "dense granule core": 6
}

# %% [markdown]
# ## Load the Pre-trained Transformer Model
# We use pre-trained DINOv2 small model for feature extraction.

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# %%
dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg").to(device)
dinov2.eval()

# %% [markdown]
# We will use the dino's `get_intermediate_layers` method to extract features from the DINOv2 model.  
# - The first parameter is an input image batch. 
# - The second parameter, `n`, points to model's layer(s) to extract features from (layers or n last layers to take).  
# - If `reshape=True`, the features will be returned as a batch of 3D : (F-size, W, H), else it will be 2D ((W x H), F-size).  
# - We don't want the class token, so `return_class_token=False`.  
# <br><br>
# This method returns a tuple of features with each element points to a requested layer.
# <br> See the code [*here*](https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L298).

# %%
help(dinov2.get_intermediate_layers)

# %%
print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

# %%
# DINOv2 trained on image patches of size 14 x 14. Therefore, the input image size should be divisible by 14. 
# dinov2_vits14_reg specs:
feature_dim = 384
patch_size = 14
# to reduce original image resolution to integer number of patches
num_patches = 30

input_size = patch_size * num_patches
print(f"Dino input image size: {input_size}")

# define proper image/mask transformation
dino_transforms = tv_transforms2.Compose([
    tv_transforms2.ToImage(),
    tv_transforms2.Resize(input_size, interpolation=tv_transforms2.InterpolationMode.BILINEAR),
    tv_transforms2.ToDtype(dtype=torch.float32, scale=True),
])

mask_transforms = tv_transforms2.Compose([
    tv_transforms2.ToImage(),
    tv_transforms2.Resize(input_size, interpolation=tv_transforms2.InterpolationMode.NEAREST)
])

# %% [markdown]
# ## Feature Extraction

# %%
# select a random batch of images and their masks
batch_size = 12
random_indices = torch.randperm(len(data_images))
image_batch = data_images[random_indices[:batch_size]]
mask_batch = gt_masks[random_indices[:batch_size]]

print(image_batch.shape)

# %%
# transform the batch for the dino model,
# also, we downscale the gt masks to the input size.
transformed_images = []
transformed_masks = []

for i in range(len(image_batch)):
    transformed_images.append(dino_transforms(image_batch[i]))
    transformed_masks.append(mask_transforms(mask_batch[i][:, :, np.newaxis]))

transformed_images = torch.stack(transformed_images).to(device)
transformed_masks = torch.stack(transformed_masks).squeeze(1)

print(transformed_images.shape, transformed_masks.shape)

# %%
# extract the features
with torch.no_grad():
    features = dinov2.get_intermediate_layers(
        transformed_images,
        n=1,
        return_class_token=False,
        reshape=False,
        norm=True
    )[0]

print(features.shape)

# %% [markdown]
# <div class="alert alert-success">
#   <h3>Checkpoint 1</h3>
#   <p>At this point we got familiar with the data, and the DINOv2 model loading and feature extraction process.</p>
# </div>

# %% [markdown]
# ## Visualization

# %% [markdown]
# <div class="alert alert-info">
#   <h3>Task 1.1: PCA on the extracted features</h3>
#   <p>
#       We want to use <i>PCA</i> as a dimensionality reduction algorithm to get first <i>3</i> principal components.<br>Then plot the outcome to compare reduced feature space with the pixel space, using those PCA components as RGB channels.
#   </p>
# <p><i>
#    Please refer to <a href="https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA">scikit-learn <b>PCA</b> documentation</a>
#</i>.</p>
# </div>

# %%
# flatten the features across all image patches (30x30)
flatten_features = features.cpu().numpy().reshape((-1, feature_dim))
print(flatten_features.shape)

# %%
# create low-res mask (30x30) to get approximate labels for each patch.
low_res_masks = F.interpolate(
    transformed_masks.unsqueeze(1),
    size=(num_patches, num_patches),
    mode="nearest-exact"
).squeeze(1)

# %%
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# %%
# PCA example:
dummy_data = np.random.rand(1000, 380)
pca = PCA(n_components=3, whiten=True, random_state=SEED)
pca_comps = pca.fit_transform(dummy_data)
pca_comps = MinMaxScaler().fit_transform(pca_comps)
print(pca_comps.shape)

# %% tags=["solution"]
# get PCA first three components. use flatten_features as input.

# insert your code here
# pca = ...
pca = PCA(n_components=3, whiten=True, random_state=SEED)
pca_comps = pca.fit_transform(flatten_features)

print(pca_comps.shape)

# %% tags=["solution"]
# scale components into range of [0, 1]
# insert your code here
pca_comps = MinMaxScaler().fit_transform(pca_comps)
print(pca_comps.min(), pca_comps.max())

# %% tags=["solution"]
# now reshape the acquired components to (batch_size, num_patches, num_patches, 3)
# insert your code here
pca_comps = pca_comps.reshape(batch_size, num_patches, num_patches, 3)
print(pca_comps.shape)

# %%
# provided function for plotting
def plot_pca(image, pca_image):
    if image.shape[0] == 3:
        image = image[0]
    image = image.cpu()

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3), layout="compressed")
    fig.canvas.toolbar_position = "right"
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False

    axes[0].imshow(image, cmap="grey", origin="lower")
    axes[0].set_title("Image")
    axes[1].imshow(pca_image, origin="lower")
    axes[1].set_title("PCA")

    for ax in axes.ravel():
        ax.set_aspect("equal", "box")
        # ax.set_axis_off()
        ax.set_yticks([])
        ax.xaxis.set_tick_params(labelsize=8)

    plt.show()

# %% tags=["solution"]
# plot some samples using plot_pca() function.
# use transformed_images as pixel images versus PCA images.
# insert your code here
# plot_pca(...)
plot_pca(transformed_images[0], pca_comps[0])

plot_pca(transformed_images[10], pca_comps[10])

# %% [markdown]
# <div class="alert alert-info">
#   <h3>Task 1.2: UMAP on the extracted features</h3>
#   <p>
#       Now, we want to reduce the dimensionality of the extracted features, and plot the reduced features using <i>UMAP</i>.
#   </p>
# <p><i>
#    Please find the documents and examples here: <a href="https://umap-learn.readthedocs.io/en/latest/parameters.html"><b>UMAP</b></a>
#</i>.</p>
# </div>

# %%
import umap

# %% tags=["solution"]
# insert your code here
# reducer = umap.UMAP(...)
reducer = umap.UMAP(n_neighbors=15, min_dist=0.3, metric="euclidean")
# use flatten_features as input
# umap_embeddings = ...
umap_embeddings = reducer.fit_transform(flatten_features)

print(umap_embeddings.shape)

# %%
# plot UMAP
fig, ax = plt.subplots(1, 1, figsize=(7, 6), layout="compressed")
fig.canvas.toolbar_position = "right"
fig.canvas.header_visible = False
fig.canvas.footer_visible = False

labels = low_res_masks.numpy().flatten()

ax.scatter(
    umap_embeddings[:, 0],
    umap_embeddings[:, 1],
    s=10, c=labels, cmap=cm, alpha=0.5, lw=0
)
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")

plt.show()

# %% [markdown]
# <div class="alert alert-success">
#   <h3>Checkpoint 2</h3>
#   <p>So far, we tried PCA and UMAP to reduce dimensionality of the extracted features for visualizing purposes.<br>As we can see, those reduced features can carry some information about the data classes and make a visually interesting representation, even though they have a low resolution.</p>
# </div>

# %% [markdown]
# ## Clustering

# %% [markdown]
# <div class="alert alert-info">
#   <h3>Task 2.1: KMeans Clustering</h3>
#   <p>
#       Finally, we want to run a KMeans clustering on the extracted features to see how an unsupervised method can perform on separating the data classes.
# In other words, we want to find out if these features contain some information about the class they belong to.
#   </p>
# <p><i>
#    You can check out <b>KMeans</b> documentation <a href="https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html">here</a>
#</i>.</p>
# </div>

# %%
from sklearn.cluster import KMeans

# %%
# clustering plot function
def plot_clustering(image, gt, gt_low, pred, cmap="Dark2", n_classes=7, clustering="KMeans"):
    if image.shape[0] == 3:
        image = image[0]
    image = image.cpu()

    fig, axes = plt.subplots(2, 2, figsize=(7, 5.9), layout="compressed")
    fig.canvas.toolbar_position = "right"
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False
    axes[0, 0].imshow(image, cmap="grey", origin="lower")
    axes[0, 0].set_title("Image")
    axes[0, 1].imshow(gt, cmap=cmap, vmax=n_classes - 1, interpolation="none", origin="lower")
    axes[0, 1].set_title("GT")
    axes[1, 0].imshow(gt_low, cmap=cmap, vmax=n_classes - 1, interpolation="none", origin="lower")
    axes[1, 0].set_title("GT (low res.)", y=-0.1, pad=2)
    axes[1, 1].imshow(pred, cmap="Set2", interpolation="none", origin="lower")
    axes[1, 1].set_title(clustering, y=-0.1, pad=0)

    for ax in axes.ravel():
        ax.set_aspect("equal", "box")
        # ax.set_axis_off()
        ax.set_xticks([])
        ax.yaxis.set_tick_params(labelsize=8)

    plt.show()

# %% tags=["solution"]
# applying KMeans: use flatten_features as input.
# set number of clusters as the same number of classes.

# insert your code here
# kmeans = ...
kmeans = KMeans(
    n_clusters=num_classes, n_init=5, max_iter=400,
    verbose=0, random_state=SEED
)

kmeans.fit(flatten_features)

# %% tags=["solution"]
# get the predictions, and un-flatten it considering the batch_size.

# insert your code here
# predictions = ...
predictions = kmeans.predict(flatten_features).reshape(-1, num_patches**2)
print(predictions.shape)

# %%
# plotting some samples
# select a sample from the batch and make it 2D
selected_idx = 0

# %% tags=["solution"]
# insert your code here
# pred_img = ...
pred_img = predictions[selected_idx].reshape(num_patches, num_patches)

# %%
plot_clustering(
    image=transformed_images[selected_idx], gt=transformed_masks[selected_idx],
    gt_low=low_res_masks[selected_idx], pred=pred_img,
    cmap=cm, n_classes=num_classes
)

# %% tags=["solution"]
# plot another sample
selected_idx = 10

# insert your code here
# pred_img = ...
pred_img = predictions[selected_idx].reshape(num_patches, num_patches)

# plot
plot_clustering(
    transformed_images[selected_idx], transformed_masks[selected_idx],
    low_res_masks[selected_idx], pred_img,
    cm, num_classes
)

# %% [markdown]
# <div class="alert alert-info">
#   <h3>Task 2.2: KMeans with different number of clusters</h3>
#   <p>
#       Try KMeans with different number of clusters and plot the results. See how it performs compares to semantic classes in the pixel space.
#   </p>

# %% tags=["solution"]
# insert your code here
kmeans = KMeans(
    n_clusters=3, n_init=5, max_iter=400,
    verbose=0, random_state=SEED
)
kmeans.fit(flatten_features)
predictions = kmeans.predict(flatten_features).reshape(-1, num_patches**2)

# %%
pred_img = predictions[0].reshape(num_patches, num_patches)
plot_clustering(
    transformed_images[0], transformed_masks[0],
    low_res_masks[0], pred_img,
    cm, num_classes
)

# %% [markdown]
# <div class="alert alert-success">
#   <h3>Checkpoint 3</h3>
#   <p>We managed to run KMeans clustering on the extracted features and visualize the resulting clusters.
# </p>
# </div>

# %% [markdown]
# #### Optional Task
# Also, as an extra optional step, you may want to use different layers of the DINO model to extract features from,
# and see the differences in PCA or Clustering results.
# You can use `dinov2.get_intermediate_layers()` function and pass a list of layers indices or a single integer (check the feature extraction cell).

# %%
