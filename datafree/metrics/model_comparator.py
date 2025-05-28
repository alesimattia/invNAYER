import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np


#from sklearn.metrics import pairwise_distances

class Comparator:
    def __init__(self, model1, model2, dataset_root='../CIFAR10', batch_size=512, num_workers=4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model1 = model1.to(self.device).eval()
        self.model2 = model2.to(self.device).eval()
        self.dataset_root = dataset_root
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        '''
            Normalizza ogni canale dell'immagine RGB usando la media e la deviazione standard calcolate sul dataset CIFAR-10.
            Ciò aiuta i modelli a convergere meglio durante l'addestramento e l'inferenza.
        '''
        self.test_dataset = datasets.CIFAR10(root=dataset_root, train=False, download=True, transform=self.transform)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        self.num_classes = len(self.test_dataset.classes)
        


    def prediction_distance(self, png=False, save_path="./distance_IMG/dist.png"):
        '''
        Calcola la NORMA MATRICIALE (Frobenius) tra le predizioni di due modelli, per ogni classe.
        - Rallenta l'esecuzione
        Params:
            - png : se True, salva un grafico delle distanze per ogni classe in save_path.
            - save_path : percorso dove salvare il grafico PNG delle distanze.
        Returns: 
            - Dizionario {classe: distanza_media}
            - Se png=True, salva un grafico delle distanze per ogni classe in save_path.
        '''

        class_distances = {i: [] for i in range(self.num_classes)}
        '''
            https://docs.pytorch.org/vision/main/datasets.html
            datasets => torchvision.datasets
            Da documentazione Pytorch:
            train=False If True, creates dataset from training set, otherwise creates from test set.
        '''
        with torch.no_grad():
            for images, targets in self.test_loader:
                images = images.to(self.device)
                model1_outputs = self.model1(images)
                model2_outputs = self.model2(images)
                distances = torch.norm(model1_outputs - model2_outputs, dim=1).cpu().numpy()
                # Raggruppa le distanze per classe
                for i, target in enumerate(targets.numpy()):
                    class_distances[target].append(distances[i])

        # Distanza MEDIA per ogni classe
        mean_distances = {class_idx: np.mean(class_distances[class_idx]) for class_idx in class_distances}
        
        if png:
            import matplotlib.pyplot as plt
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            class_labels = list(mean_distances.keys())
            distances = [mean_distances[k] for k in class_labels]
            plt.figure(figsize=(8, 5))
            plt.bar(class_labels, distances, color='skyblue')
            plt.xlabel("Classe")
            plt.ylabel("Distanza media")
            plt.title(f"Distanza media tra le predizioni del modello {self.model1} e {self.model2} per ogni classe")
            plt.xticks(class_labels)
            plt.savefig(save_path)
            plt.close()

        return mean_distances



    def dice_coefficient(self):
        '''
        Calcola il coefficiente di Dice per ogni classe tra le predizioni di due modelli.
            Returns: dizionario {classe: dice_score}
        '''
        preds1, preds2, groundT = [], [], []

        with torch.no_grad():
            for images, targets in self.test_loader:
                images = images.to(self.device)
                model1_outputs = self.model1(images)
                model2_outputs = self.model2(images)
                preds1.append(torch.argmax(model1_outputs, dim=1).cpu().numpy())
                preds2.append(torch.argmax(model2_outputs, dim=1).cpu().numpy())
                groundT.append(targets.numpy())

        preds1 = np.concatenate(preds1)
        preds2 = np.concatenate(preds2)
        groundT = np.concatenate(groundT)
        '''
        Usa le etichette reali (groundT) per selezionare solo le immagini della classe corrente. 
        '''
        dice_scores = {}
        for currentClass in range(self.num_classes):
            idx = (groundT == currentClass)
            if np.sum(idx) == 0:
                dice_scores[currentClass] = np.nan
                continue
            pred1 = preds1[idx]
            pred2 = preds2[idx]
            intersection = np.sum(pred1 == pred2)                #evita divisione per zero
            dice = (2. * intersection) / (len(pred1) + len(pred2) + 1e-8)
            dice_scores[currentClass] = dice

        return dice_scores
    
    
    def jensen_Shannon_index(self):
        """
        Calcola l'indice Jensen-Shannon medio tra le predizioni (softmax) di due modelli su un dataset.
        Restituisce il valore medio su tutte le immagini.
        """
        from scipy.spatial.distance import jensenshannon
        js_distances = []

        with torch.no_grad():
            for images, _ in self.test_loader:
                images = images.to(self.device)
                out1 = torch.softmax(self.model1(images), dim=1).cpu().numpy()
                out2 = torch.softmax(self.model2(images), dim=1).cpu().numpy()
                
                # Calcola JS per ogni immagine del batch
                for p, q in zip(out1, out2):
                    js = jensenshannon(p, q, base=2)  # base=2 per JS index in [0,1]
                    js_distances.append(js)

        js_mean = np.mean(js_distances)
        return js_mean


    def plot_decision_boundary(self, model, dataset, device=None, batch_size=512, num_workers=4, output_path="decision_boundary.png"):
        """
        Calcola e visualizza i decision boundary delle predizioni del modello.
        - Le feature vengono ridotte a 2D tramite PCA.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        #from sklearn.decomposition import PCA

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        features = []
        labels = []
        with torch.no_grad():
            for images, targets in loader:
                images = images.to(device)
                # Estrai le feature dallo strato penultimo (ResNet: return_features=True)
                outputs, feats = model(images, return_features=True)
                features.append(feats.cpu().numpy())
                labels.append(targets.numpy())
        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)

        # Riduci a 2D con PCA
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)

        # Crea una griglia per il plot dei confini
        x_min, x_max = features_2d[:, 0].min() - 1, features_2d[:, 0].max() + 1
        y_min, y_max = features_2d[:, 1].min() - 1, features_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
        grid = np.c_[xx.ravel(), yy.ravel()]

        # Proietta la griglia nello spazio originale delle feature
        grid_original = pca.inverse_transform(grid)
        grid_tensor = torch.tensor(grid_original, dtype=torch.float32).to(device)
        with torch.no_grad():
            # Passa le feature direttamente al classificatore lineare
            logits = model.linear(grid_tensor)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        preds = preds.reshape(xx.shape)

        # Plot
        plt.figure(figsize=(10, 8))
        cmap = ListedColormap(plt.cm.tab10.colors[:len(np.unique(labels))])
        plt.contourf(xx, yy, preds, alpha=0.3, cmap=cmap)
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap=cmap, edgecolor='k', s=20)
        plt.xlabel("PCA1")
        plt.ylabel("PCA2")
        plt.title("Decision Boundary del modello ResNet18")
        plt.legend(*scatter.legend_elements(), title="Classi")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Decision boundary salvato in: {output_path}")