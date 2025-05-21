### IMPORTS
import pickle
from sklearn.datasets import fetch_lfw_pairs
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
from torch.utils.data import DataLoader, SequentialSampler
from torchvision import datasets, transforms
import numpy as np
import os

data_home = os.getcwd()
print(data_home)


# Fetch LFW pairs dataset
fetch_lfw_pairs(subset="10_folds", data_home=data_home, funneled=False, resize=1, color=True)
print("Downlaoded dataset")

data_dir = data_home + "/lfw_home/lfw"
pairs_dir = data_home + "/lfw_home/pairs.txt"


# Set up parameters
batch_size = 16
epochs = 15
workers = 0 if os.name == 'nt' else 2
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(
    image_size=160,
    margin=14,
    device=device,
    selection_method='center_weighted_size'
)

# Define the data loader for the input set of images
orig_img_ds = datasets.ImageFolder(data_dir, transform=None)

# overwrites class labels in dataset with path so path can be used for saving output in mtcnn batches
orig_img_ds.samples = [
    (p, p)
    for p, _ in orig_img_ds.samples
]

loader = DataLoader(
    orig_img_ds,
    num_workers=workers,
    batch_size=batch_size,
    collate_fn=training.collate_pil
)


crop_paths = []
box_probs = []

for i, (x, b_paths) in enumerate(loader):
    crops = [p.replace(data_dir, data_dir + '_cropped') for p in b_paths]
    for crop_path in crops:
        os.makedirs(os.path.dirname(crop_path), exist_ok=True)
    mtcnn(x, save_path=crops)
    crop_paths.extend(crops)
    print('\rBatch {} of {}'.format(i + 1, len(loader)), end='')

print()
print("Saved to cropped paths")

# Remove mtcnn to reduce GPU memory usage
del mtcnn
torch.cuda.empty_cache()

# create dataset and data loaders from cropped images output from MTCNN
trans = transforms.Compose([
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization
])

dataset = datasets.ImageFolder(data_dir + '_cropped', transform=trans)

embed_loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    sampler=SequentialSampler(dataset)
)

# Load pretrained resnet model
resnet = InceptionResnetV1(
    classify=False,
    pretrained='vggface2'
).to(device)

classes = []
embeddings = []
resnet.eval()
with torch.no_grad():
    for xb, yb in embed_loader:
        xb = xb.to(device)
        b_embeddings = resnet(xb)
        b_embeddings = b_embeddings.to('cpu').numpy()
        classes.extend(yb.numpy())
        embeddings.extend(b_embeddings)

embeddings_dict = dict(zip(crop_paths,embeddings))

## Convert lists to NumPy arrays
embeddings = np.array(embeddings)
classes = np.array(classes)

## Save to a .npz file
np.savez("embeddings_ceci.npz", embeddings=embeddings, classes=classes)

with open("embeddings_dict_ceci.pkl", "wb") as f:
    pickle.dump(embeddings_dict, f)
print("Embeddings and dict saved successfully!")
