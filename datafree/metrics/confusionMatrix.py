import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
import numpy as np

def compute_confusion_matrix(model, dataset_root, output_path="./CM.png", batch_size=512, num_workers=4):
    '''
    Returns: grafico matplotlib della matrice di confusione tra le predizioni del modello e le etichette reali del dataset.
	'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    '''
        Normalizza ogni canale dell'immagine RGB usando la media e la deviazione standard calcolate sul dataset CIFAR-10.
        Ciò aiuta i modelli a convergere meglio durante l'addestramento e l'inferenza.
    '''
    
    dataset = datasets.CIFAR10(root=dataset_root, train=False, download=False, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    cm = confusion_matrix(all_targets, all_preds)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(dataset.classes)),
        yticks=np.arange(len(dataset.classes)),
        xticklabels=dataset.classes,
        yticklabels=dataset.classes,
        ylabel='True label',
        xlabel='Predicted label',
        title=f'{model} Confusion Matrix'
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(output_path)
    
    return fig