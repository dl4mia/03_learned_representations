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
# <h2 style="font-family: serif">II. Part Two</h2>

# %% [markdown]
# In the second part, we will train a model using the DINOv2 extracted features as inputs. The task will be semantic segmentation over the input image patches.
# <br>For model evaluation, we are using metrics from [**Segmentation Models**](https://smp.readthedocs.io/en/latest/metrics.html) package.

# %%
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as tv_transforms2

from tqdm.notebook import tqdm

import utils
import metrics

# %%
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

# %%
# plotting functions

def plot_metric(metric, ax, title, colors, class_labels):
    # metric shape: num_epochs x num_classes
    num_classes = metric.shape[1]
    for c in range(num_classes):
        ax.plot(metric[:, c] * 100, color=colors[c], label=class_labels[c], lw=1.2)
    # epoch average
    ax.plot(metric.mean(axis=1) * 100, color="maroon", label="Average", lw=1.5)

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("%")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)


def plot_evaluation(metric, ax, title, colors, class_labels):
    bp = ax.bar(class_labels, height=metric * 100, color=colors, width=0.65)
    ax.bar_label(bp, label_type="edge", fmt="%.2f")
    ax.set_title(title, fontweight="bold")
    ax.tick_params(axis="x", labelrotation=45)


# %% [markdown]
# ## Dataset
# We are using data provided by [*Dense cellular segmentation for EM using 2D-3D neural network ensembles*](https://leapmanlab.github.io/dense-cell/).
# <br>The data contains *tiff* files in train and evaluation sets along with their ground truth masks. Masks include dense annotations for seven classes.
# <br>Images have a resolution of 800 x 800 pixels which are a bit large to fit in memory or GPU 😁 . However, we'll transform them into a smaller manageable resolution.

# %%
# dataset class definition

class TiffDataset(Dataset):
    def __init__(self, image_path, label_path, input_size, train=True):
        self.images = utils.get_images_from_tiff(image_path, to_rgb=True)  # numpy array, channel last
        self.gt_masks = utils.get_images_from_tiff(label_path, to_rgb=False)
        self.input_size = input_size
        self.train = train
        self.img_h = self.images.shape[1]
        self.img_w = self.images.shape[2]
        self.mean = None
        self.std = None
        self.mean, self.std = self.get_mean_std()
        self.base_transform = tv_transforms2.Compose([
            tv_transforms2.ToImage(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        mask = self.gt_masks[index]

        return self.apply_transform(image, mask)

    def apply_transform(self, image, mask):
        # check channel dimension
        if len(image.shape) < 3:
            image = np.expand_dims(image, axis=-1)
        if len(mask.shape) < 3:
            mask = np.expand_dims(mask, axis=-1)

        image, mask = self.base_transform(image, mask)
        # resizing
        image = tv_transforms2.functional.resize(image, size=self.input_size)
        mask = tv_transforms2.functional.resize(
            mask, size=self.input_size,
            interpolation=tv_transforms2.InterpolationMode.NEAREST_EXACT, antialias=False
        )
        # to tensor
        image = tv_transforms2.functional.to_dtype(image, dtype=torch.float32, scale=True)
        mask = tv_transforms2.functional.to_dtype(mask, dtype=torch.long, scale=False)
        # normalizing
        image = tv_transforms2.functional.normalize(image, self.mean, self.std)
        assert torch.isin(torch.unique(mask), torch.arange(7)).all()

        return image, mask.squeeze(0)

    def get_mean_std(self):
        _min = self.images.min()
        _max = self.images.max()
        scaled_imgs = (self.images - _min) / (_max - _min)
    
        return scaled_imgs.mean(), scaled_imgs.std()

    def get_class_weights(self):
        _, weights = np.unique(self.gt_masks, return_counts=True)
        weights = 1 / weights
        # normalize weights
        weights = weights / weights.sum()

        return weights

    def plot_sample(self, cm=None):
        image, mask = self.__getitem__(0)
        cm = cm or "Dark2"
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.0), layout="compressed")
        fig.canvas.toolbar_position = "right"
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
        axes[0].imshow(image[0], cmap="grey")
        axes[0].set_title("Image")
        axes[1].imshow(mask, cmap=cm, vmax=7, interpolation="none")
        axes[1].set_title("Label")
        axes[2].imshow(image[0], interpolation="none", cmap="grey")
        axes[2].imshow(mask, alpha=0.45, cmap=cm, vmax=7, interpolation="none")
        axes[2].set_title("Overlay")

        plt.show()

# %%
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

# %% [markdown]
# ## Segmentation Model

# %% [markdown]
# <div class="alert alert-info">
#   <h3>Task 1.1: Implementing a Model for Segmentation</h3>
#   <p>
#       It's time to define our segmentation model!
#       <br>Start with a simple model, use convolution layers, and remember that the input has a resolution of (num_patches × num_patches).
#   </p>
# <p><i>
#    Please refer to the <a href="https://pytorch.org/docs/stable/nn.html"><b>Pytorch</b> documentation</a>
#</i>.</p>
# </div>


# %%
# the base class for your model to derive from (just gives you the number of trainable params) :)
class BaseNet(nn.Module):
    def __init__(self):
        super().__init__()

    def count_parameters(self, trainable=True):
        params = [
            param.numel()
            for param in self.parameters() if param.requires_grad == trainable
        ]
        return sum(params), params

    def __repr__(self):
        params = self.count_parameters()
        return f"{super().__repr__()}\ntrainable params: {params[0]:,d} {params[1]}"

# %% tags=["solution"]
# complete the model implementation

# class Net(BaseNet):
#     def __init__(self, ...):
#         super().__init__()
#         self.input_dim = num_patches
#         self.in_channels = in_channels
#         self.n_classes = n_classes
#         
#         self.conv1 = nn.Sequential(
#              nn.Conv2d(
#                   self.in_channels, 256, kernel_size=3, padding=1, bias=False
#               ),
#              nn.BatchNorm2d(256),
#              nn.LeakyReLU(negative_slope=0.01),
#         )
#        # add a similar module as above. Note that input channels to this module will be the same as what is output channels
#        self.conv2 = ...
#        # add a similar module. Note that output channels of this should be same as input channels of self.conv_out
#        self.conv3 = ...
#        # segmentation output should have channels equal to the number of classes.        
#        self.conv_out = nn.Conv2d(
#            64, self.n_classes, kernel_size=1, bias=False
#        )
#
#    def forward(self, x):
#        # input will be a tensor of b x (num_patches^2) x 384 dims.
#        x = x.reshape(-1, self.input_dim, self.input_dim, self.in_channels)
#        x = x.permute(0, 3, 1, 2)
#        out = self.conv1(x)
#        out = self.conv2(out)
#        out = self.conv3(out)
#        out = self.conv_out(out)

#        return out

class Net(BaseNet):
    def __init__(self, num_patches, in_channels, n_classes):
        super().__init__()
        self.input_dim = num_patches
        self.in_channels = in_channels
        self.n_classes = n_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                self.in_channels, 256, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                256, 128, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                128, 64, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.conv_out = nn.Conv2d(
            64, self.n_classes, kernel_size=1, bias=False
        )

    def forward(self, x):
        x = x.reshape(-1, self.input_dim, self.input_dim, self.in_channels)
        x = x.permute(0, 3, 1, 2)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv_out(out)

        return out

# %%
# DINOv2 trained on image patches of size 14 x 14. Therefore, the input image size should be divisible by 14. 
# dinov2_vits14_reg specs:
feature_dim = 384
patch_size = 14
# to reduce original image resolution to integer number of patches
num_patches = 30

input_size = patch_size * num_patches
print(f"Dino input image size: {input_size}")


# %%
model = Net(num_patches, feature_dim, num_classes).to(device)
print(model)

# %%
print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

# %% [markdown]
# ## Training

# %% [markdown]
# <div class="alert alert-info">
#   <h3>Task 1.2: Training</h3>
#   <p>
#       The training loop is here already! You just need to setup the optimizer and the loss function.
#       <br><b>Note:</b> In segmentation tasks, usually some classes including much more pixels than others e.g. background. So, we need to take care of this class imbalance giving weights to each classes. To do this use <code>train_dataset.get_class_weights()</code> to get the class's weights.
#   </p>
# <p><i>
#    Please refer to the <a href="https://pytorch.org/docs/stable/nn.html"><b>Pytorch</b> documentation</a>
#</i>.</p>
# </div>
# %%
# the data resides in this path: "/group/dl4miacourse/platelet_data"
# train dataset
train_dataset = TiffDataset(
    image_path="/group/dl4miacourse/platelet_data/train-images.tif",
    label_path="/group/dl4miacourse/platelet_data/train-labels.tif",
    input_size=input_size
)

print(f"number of images: {len(train_dataset)}")
train_dataset.plot_sample(cm)

# %%
# hyper-params
batch_size = 16
lr = 1e-3
epochs = 10

# %% tags=["solution"]
# insert your code here
# optim = ...  # you can use Adam or AdamW
optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.03)

# class_weights = torch.from_numpy(train_dataset.get_class_weights()).to(torch.float32).to(device)
# loss_fn = ...  # use CrossEntropyLoss with the weight param equals to the class weights
class_weights = torch.from_numpy(train_dataset.get_class_weights()).to(torch.float32).to(device)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

# %%
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
num_batches = len(train_loader)
print(f"number of batches: {num_batches}")

# %%
# preparing training loss plot
fig_loss, ax_loss = plt.subplots(1, 1, figsize=(11, 6), layout="compressed")
fig_loss.canvas.toolbar_position = "right"
fig_loss.canvas.header_visible = False
fig_loss.canvas.footer_visible = False
ax_loss.set_title("Training Loss")
ax_loss.set_xlabel("Iterations")
ax_loss.set_ylabel("Loss")
ax_loss.grid(alpha=0.3)
loss_line = None

# %%
plt.show()
losses = []
tps, fps, fns, tns = [], [], [], []

for e in tqdm(range(epochs), desc="Training Epochs"):
    for batch_idx, (image, gt_masks) in enumerate(train_loader):
        image = image.to(device)
        gt_masks = gt_masks.to(device)

        with torch.no_grad():
            features = dinov2.get_intermediate_layers(image, 1, return_class_token=False)[0]
        out = model(features)

        # here we scale up the model output into the gt_mask size
        out_upscaled = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)

        loss = loss_fn(out_upscaled, gt_masks)
        optim.zero_grad()
        loss.backward()
        optim.step()

        # for metrics
        tp, fp, fn, tn = metrics.get_stats(
            out_upscaled.argmax(dim=1),
            gt_masks, mode="multiclass",
            num_classes=num_classes
        )
        tps.append(tp)
        fps.append(fp)
        fns.append(fn)
        tns.append(tn)

        losses.append(loss.item())
        if batch_idx == 0 or batch_idx % 5 == 0:
            print(f"loss: {loss.item():.5f}", end="\r")
            if loss_line:
                loss_line[0].remove()
            loss_line = ax_loss.plot(losses, color="dodgerblue", label="Train Loss")
            fig_loss.canvas.draw()
            fig_loss.canvas.flush_events()
            ax_loss.legend()
    # end of one epoch

# %%
# calculate metrics
accs = []
f1_scores = []
ious = []

for i in range(0, len(tps), num_batches):
    epoch_tp = torch.concat(tps[i: i + num_batches])
    epoch_fp = torch.concat(fps[i: i + num_batches])
    epoch_fn = torch.concat(fns[i: i + num_batches])
    epoch_tn = torch.concat(tns[i: i + num_batches])
    accs.append(
        metrics.accuracy(epoch_tp, epoch_fp, epoch_fn, epoch_tn, reduction=None).mean(dim=0).numpy()
    )
    f1_scores.append(
        metrics.f1_score(epoch_tp, epoch_fp, epoch_fn, epoch_tn, reduction=None).mean(dim=0).numpy()
    )
    ious.append(
        metrics.iou_score(epoch_tp, epoch_fp, epoch_fn, epoch_tn, reduction=None).mean(dim=0).numpy()
    )

accs = np.vstack(accs)
f1_scores = np.vstack(f1_scores)
ious = np.vstack(ious)

# %%
# plot metrics
fig_metrics, axes = plt.subplots(1, 3, figsize=(16, 5), layout="compressed")
fig_metrics.canvas.toolbar_position = "right"
fig_metrics.canvas.header_visible = False
fig_metrics.canvas.footer_visible = False

plot_metric(accs, axes[0], "Accuracy", colors, list(classes.keys()))
plot_metric(f1_scores, axes[1], "F1 Score", colors, list(classes.keys()))
plot_metric(ious, axes[2], "IoU", colors, list(classes.keys()))

plt.show()

# %% [markdown]
# <div class="alert alert-success">
#   <h3>Checkpoint 1</h3>
#   <p>Congratulations! You managed to train a segmentation model using the DINOv2 features as inputs.<br>Let's evaluate your model! 😁</p>
# </div>

# %% [markdown]
# ## Evaluation

# %% [markdown]
# <div class="alert alert-info">
#   <h3>Task 1.3: Model Evaluation</h3>
#   <p>
#       Given the train loop, this is a very easy one!
#       <br>You need to pass the input image to the DINOv2 to get the features, and then pass the features to your model. Done!
#   </p>
# </div>

# %%
# test dataset 
# Note: we are using normalization stats from the train dataset.
mean = train_dataset.mean
std = train_dataset.std

# the data resides in this path: "/group/dl4miacourse/platelet_data"
test_dataset = TiffDataset(
    image_path="/group/dl4miacourse/platelet_data/eval-images.tif",
    label_path="/group/dl4miacourse/platelet_data/eval-labels.tif",
    input_size=input_size, train=False
)
test_dataset.mean = mean
test_dataset.std = std

test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

print(f"number of images: {len(test_dataset)}")

# %%
losses = []
tps, fps, fns, tns = [], [], [], []

model.eval()

# %% tags=["solution"]
# complete the testing code
# for image, gt_masks in test_loader:
#    image = image.to(device)
#    gt_masks = gt_masks.to(device)

#    with torch.no_grad():
#        # pass image to the DINO to get the features
#        features = ...
#        # pass features to your model
#        out = ...
#
#    out_upscaled = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)
#    loss = loss_fn(out_upscaled, gt_masks)
#    Don't forget your metrics!
#    tp, fp, fn, tn = metrics.get_stats(
#        out_upscaled.argmax(dim=1),
#        gt_masks, mode="multiclass",
#        num_classes=num_classes
#    )
#    tps.append(tp)
#    fps.append(fp)
#    fns.append(fn)
#    tns.append(tn)
#    losses.append(loss.item())
#
# print(f"Evaluation average loss: {np.mean(losses):.5f}", end="\r")

for image, gt_masks in test_loader:
    image = image.to(device)
    gt_masks = gt_masks.to(device)

    with torch.no_grad():
        features = dinov2.get_intermediate_layers(image, 1, return_class_token=False)[0]
        out = model(features)

    out_upscaled = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)
    loss = loss_fn(out_upscaled, gt_masks)

    tp, fp, fn, tn = metrics.get_stats(
        out_upscaled.argmax(dim=1),
        gt_masks, mode="multiclass",
        num_classes=num_classes
    )
    tps.append(tp)
    fps.append(fp)
    fns.append(fn)
    tns.append(tn)

    losses.append(loss.item())

print(f"Evaluation average loss: {np.mean(losses):.5f}", end="\r")

# %%
# calculate metrics
tps = torch.concat(tps)
fps = torch.concat(fps)
fns = torch.concat(fns)
tns = torch.concat(tns)

accs = metrics.accuracy(tps, fps, fns, tns, reduction=None).mean(dim=0).numpy()
f1_scores = metrics.f1_score(tps, fps, fns, tns, reduction=None).mean(dim=0).numpy()
ious = metrics.iou_score(tps, fps, fns, tns, reduction=None).mean(dim=0).numpy()

# %%
# plot evaluation metrics
fig_metrics, axes = plt.subplots(1, 3, figsize=(16, 5), layout="compressed")
fig_metrics.canvas.toolbar_position = "right"
fig_metrics.canvas.header_visible = False
fig_metrics.canvas.footer_visible = False

plot_evaluation(accs, axes[0], "Accuracy", colors, list(classes.keys()))
plot_evaluation(f1_scores, axes[1], "F1 Score", colors, list(classes.keys()))
plot_evaluation(ious, axes[2], "IoU", colors, list(classes.keys()))

plt.show()

# %%
# plot a sample of model's segmentation result
img, gt = test_dataset.__getitem__(0)
img = img.unsqueeze(0).to(device)

with torch.no_grad():
    features = dinov2.get_intermediate_layers(img, 1, return_class_token=False)[0]
    out = model(features)
out_upscaled = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)
pred = out_upscaled.squeeze(0)
pred = pred.argmax(dim=0)

fig, axes = plt.subplots(1, 2, figsize=(11, 6), layout="compressed")
fig.canvas.header_visible = False
axes[0].imshow(img[0, 0].cpu(), cmap="grey", interpolation="none")
axes[0].imshow(pred.cpu(), cmap=cm, interpolation="none", vmax=num_classes - 1, alpha=0.5)
axes[0].set_title("Prediction")
axes[1].imshow(img[0, 0].cpu(), cmap="grey", interpolation="none")
axes[1].imshow(gt, cmap=cm, interpolation="none", vmax=num_classes - 1, alpha=0.5)
axes[1].set_title("Ground Truth")

plt.show()

# %% [markdown]
# <div class="alert alert-success">
#   <h3>Checkpoint 2</h3>
#   <p>Congratulations × 2 !! <br>Now we learned that DINOv2 or in general vision transformer models, can extract meaningful features out of our dataset images, even though they are usually trained on natural images.
#   <br>We can use those features for many downstream tasks, including semantic segmentation. 
#   </p>
# </div>

# %%
