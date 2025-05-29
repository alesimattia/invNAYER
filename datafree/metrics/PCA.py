import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def model_PCA(model, components=3, dataset_root='../CIFAR10', save_as_png=False, batch_size=512, num_workers=4, output_path='./PCA_plot.png'):
    """
    Estrae le 3 caratteristiche principali del modello mediante PCA e le 
    visualizza in uno spazio 3D

    Memorizza su FS il grafico delle 3 componenti principali
    """
    from mpl_toolkits.mplot3d import Axes3D

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # Normalizzazione standard CIFAR-10
    ])          
            
    '''
        datasets => torchvision.datasets
        Da documentazione Pytorch:
        train=False If True, creates dataset from training set, otherwise creates from test set.
    '''
    dataset = datasets.CIFAR10(root=dataset_root, train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Estrazione delle caratteristiche
    features = []
    labels = []
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            outputs = model(images, return_features=True)[1]  # Layer Fully connected resnet.py:96
            features.append(outputs.cpu().numpy())
            labels.append(targets.numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    pca = PCA(n_components=components)
    features_pca = pca.fit_transform(features)
    colors = plt.cm.get_cmap("tab10", len(dataset.classes))

    if components == 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        for class_idx in range(len(dataset.classes)):
            class_points = features_pca[labels == class_idx]
            ax.scatter(class_points[:, 0], class_points[:, 1], label=f'Class {class_idx}', alpha=0.7, s=20, c=[colors(class_idx)])
        ax.set_xlabel("PCA1")
        ax.set_ylabel("PCA2")
    else:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for class_idx in range(len(dataset.classes)):
            class_points = features_pca[labels == class_idx]
            ax.scatter(class_points[:, 0], class_points[:, 1], class_points[:, 2], label=f'Class {class_idx}', alpha=0.7, s=20, c=[colors(class_idx)])
        ax.set_xlabel("PCA1")
        ax.set_ylabel("PCA2")
        ax.set_zlabel("PCA3")
    ax.legend()


    if save_as_png and output_path is not None:
        plt.savefig(output_path)
        print(f"PCA {components}D plot salvato in: {output_path}")
        plt.close(fig)
        return fig
    return features_pca, labels, pca


def plot_decision_boundary(model, dataset_root='../CIFAR10', batch_size=512, num_workers=4, output_path="./PCA_plot/decision_boundary.png"):
    from matplotlib.colors import ListedColormap

    # Ottieni features penultime, labels e la PCA fitted
    features_pca, labels, pca = model_PCA(
        model, components=2, dataset_root=dataset_root,
        save_as_png=False, batch_size=batch_size, num_workers=num_workers, output_path=None
    )

    # Crea una griglia nello spazio PCA 2D
    x_min, x_max = features_pca[:, 0].min() - 1, features_pca[:, 0].max() + 1
    y_min, y_max = features_pca[:, 1].min() - 1, features_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Proietta la griglia nello spazio originale delle feature penultime (es: 512-dim)
    grid_original = pca.inverse_transform(grid)  # shape: [90000, 512]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid_tensor = torch.tensor(grid_original, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model.linear(grid_tensor)  # shape: [90000, num_classes]
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    preds = preds.reshape(xx.shape)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = ListedColormap(plt.cm.tab10.colors[:len(np.unique(labels))])
    ax.contourf(xx, yy, preds, alpha=0.3, cmap=cmap)
    scatter = ax.scatter(features_pca[:, 0], features_pca[:, 1], c=labels, cmap=cmap, edgecolor='k', s=20)
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.set_title(f"Decision Boundary - {model.__class__.__name__}")
    ax.legend(*scatter.legend_elements(), title="Classi")

    if output_path is not None:
        plt.savefig(output_path)
        print(f"Decision boundary salvato in: {output_path}")

    return fig