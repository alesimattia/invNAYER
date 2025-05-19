def model_PCA(model, components=3, dataset_root='../CIFAR10', batch_size=512, num_workers=4, output_path='./PCA_plot.png'):
    """
    Estrae le 3 caratteristiche principali del modello mediante PCA e le 
    visualizza in uno spazio 3D

    Memorizza su FS il grafico delle 3 componenti principali
    """

    import torch
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

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
    cifar_test = datasets.CIFAR10(root=dataset_root, train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(cifar_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Estrazione delle caratteristiche
    features = []
    labels = []
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            outputs = model(images, return_features=True)[1]  # Classificazione
            features.append(outputs.cpu().numpy())
            labels.append(targets.numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)


    pca = PCA(n_components=components)
    features_pca = pca.fit_transform(features)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.get_cmap("tab10", 10)

    for class_idx in range(10):
        class_points = features_pca[labels == class_idx]
        ax.scatter(class_points[:, 0], class_points[:, 1], class_points[:, 2], label=f'Class {class_idx}', alpha=0.7, s=20, c=[colors(class_idx)])

    ax.set_title(f"PCA delle 3 caratteristiche principali di {model}")
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.set_zlabel("PCA3")
    ax.legend()

    plt.savefig(output_path)
    print(f"PCA 3Dplot salvato in: {output_path}")
    plt.close(fig)

    return fig