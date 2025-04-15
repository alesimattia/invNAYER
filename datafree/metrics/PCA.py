def model_pca(model, dataset_root='./CIFAR10', batch_size=512, num_workers=4, output_path='./PCA_plot.png'):

    import torch
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    # Configurazione del dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Caricamento del dataset CIFAR-10 (test set)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # Normalizzazione standard CIFAR-10
    ])
    cifar_test = datasets.CIFAR10(root=dataset_root, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(cifar_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Estrazione delle caratteristiche
    features = []
    labels = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            outputs = model(images)  # Classificazione con il modello model
            features.append(outputs.cpu().numpy())
            labels.append(targets.numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    # Riduzione della dimensionalità con PCA
    pca = PCA(n_components=3)
    features_pca = pca.fit_transform(features)

    # Creazione del grafico 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Mappatura dei colori per le classi
    colors = plt.cm.get_cmap("tab10", 10)

    for class_idx in range(10):
        class_points = features_pca[labels == class_idx]
        ax.scatter(class_points[:, 0], class_points[:, 1], class_points[:, 2], label=f'Class {class_idx}', alpha=0.7, s=20, c=[colors(class_idx)])

    ax.set_title("PCA delle caratteristiche estratte dal model (CIFAR-10)")
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.set_zlabel("PCA3")
    ax.legend()

    # Salvataggio del grafico come immagine PNG
    plt.savefig(output_path)
    print(f"Grafico salvato in: {output_path}")
    plt.close(fig)