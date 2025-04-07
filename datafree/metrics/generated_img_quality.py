import torch
import torch.nn.functional as F
from torchvision.models import inception_v3
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
from scipy.stats import entropy
import numpy as np
import os

class ImageFolderDataset(Dataset):
    def __init__(self, image_dir, transform):
        self.image_paths = [
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(img)


def inception_score_from_folder(image_dir, batch_size=32, splits=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    dataset = ImageFolderDataset(image_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.eval()

    preds = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calcolo predizioni Inception"):
            batch = batch.to(device)
            logits = model(batch)
            softmax = F.softmax(logits, dim=1)
            preds.append(softmax.cpu().numpy())

    preds = np.concatenate(preds, axis=0)  # shape (N, 1000)

    # Inception Score suddividi in splits
    split_scores = []
    N = preds.shape[0]
    for k in range(splits):
        part = preds[k * N // splits: (k+1) * N // splits]
        py = np.mean(part, axis=0)  # media marginale
        scores = [entropy(pyx, py) for pyx in part]
        split_scores.append(np.exp(np.mean(scores)))

    mean_is = np.mean(split_scores)
    std_is = np.std(split_scores)
    print(f"Inception Score: {mean_is:.4f} ± {std_is:.4f}")
    return mean_is, std_is