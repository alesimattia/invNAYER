import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def compute_TSNE(model, dataset_root, print_tag="model", batch_size=512, num_workers=4, output_path="./tsne.png"):
    
    """
    Applica t-SNE alle predizioni del modello specificato.
	- Il numero massimo di componenti è 2 da documentazione repo. (https://github.com/CannyLab/tsne-cuda)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    dataset = datasets.CIFAR10(root=dataset_root, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    features = []
    labels = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            outputs = model(images)
            features.append(outputs.cpu().numpy())
            labels.append(targets.numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    tsne = TSNE(n_components=2, random_state=42)
    features_tsne = tsne.fit_transform(features)


    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.get_cmap("tab10", 10)
    for class_idx in range(10):
        idxs = labels == class_idx
        ax.scatter(features_tsne[idxs, 0], features_tsne[idxs, 1], label=dataset.classes[class_idx], alpha=0.7, s=20, c=[colors(class_idx)])
    ax.legend()
    ax.set_title(f"t-SNE predizioni - {print_tag}")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    plt.tight_layout()
    plt.savefig(output_path)
    
    print(f"TSNE plot salvato in: {output_path}")