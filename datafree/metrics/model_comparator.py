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

        '''
            Normalizza ogni canale dell'immagine RGB usando la media e la deviazione standard calcolate sul dataset CIFAR-10.
            Ciò aiuta i modelli a convergere meglio durante l'addestramento e l'inferenza.
        '''
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        '''
            https://docs.pytorch.org/vision/main/datasets.html
            datasets => torchvision.datasets
            Da documentazione Pytorch:
            train=False If True, creates dataset from training set, otherwise creates from test set.
        '''
        self.test_dataset = datasets.CIFAR10(root=dataset_root, train=False, download=True, transform=self.transform)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        self.num_classes = len(self.test_dataset.classes)
        


    def prediction_distance(self, save_path=None):
        '''
        Calcola la NORMA MATRICIALE (Frobenius) tra le predizioni di due modelli, per ogni classe.
        - Rallenta l'esecuzione
        Params:
            - save_path : percorso dove salvare il grafico PNG delle distanze.
        Returns: 
            - Dizionario {classe: distanza_media}
            - Se save_path è valorizzato, esporta il grafico .PNG delle distanze per ogni classe.
        '''

        class_distances = {i: [] for i in range(self.num_classes)}
        
        with torch.no_grad():
            for images, targets in self.test_loader:
                images = images.to(self.device)
                model1_prob = torch.softmax(self.model1(images), dim=1)
                model2_prob = torch.softmax(self.model2(images), dim=1)
                distances = torch.norm(model1_prob - model2_prob, dim=1).cpu().numpy()
                # Raggruppa le distanze per classe
                for i, target in enumerate(targets.numpy()):
                    class_distances[target].append(distances[i])

        # Distanza MEDIA per ogni classe
        mean_distances = {class_idx: np.mean(class_distances[class_idx]) for class_idx in class_distances}
        
        if save_path:
            import matplotlib.pyplot as plt
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            class_labels = list(mean_distances.keys())
            distances = [mean_distances[k] for k in class_labels]
            
            plt.figure(figsize=(10, 6))
            plt.bar(class_labels, distances, color='skyblue')
            plt.xlabel("Classe")
            plt.ylabel("Distanza media")
            plt.title(f"Distanza media tra le predizioni del modello {self.model1.__class__.__name__} e {self.model2.__class__.__name__} per ogni classe")
            plt.xticks(class_labels)
            plt.savefig(save_path)
            plt.close()
            
        return mean_distances



    def dice_coefficient(self, save_path=None):
        '''
        Calcola il coefficiente di DICE per ogni classe tra le predizioni di due modelli.
        Args:
            save_path: percorso dove salvare l'istogramma PNG
        Returns: 
            dict: dizionario {classe: dice_score}
        '''
        preds1, preds2, groundT = [], [], []

        with torch.no_grad():
            for images, targets in self.test_loader:
                images = images.to(self.device)
                model1_prob = torch.softmax(self.model1(images), dim=1)
                model2_prob = torch.softmax(self.model2(images), dim=1)
                preds1.append(torch.argmax(model1_prob, dim=1).cpu().numpy())
                preds2.append(torch.argmax(model2_prob, dim=1).cpu().numpy())
                groundT.append(targets.numpy())

        preds1 = np.concatenate(preds1)
        preds2 = np.concatenate(preds2)
        groundT = np.concatenate(groundT)

        # Calcola DICE score per ogni classe
        dice_scores = {}
        for currentClass in range(self.num_classes):
            idx = (groundT == currentClass)
            if np.sum(idx) == 0:
                dice_scores[currentClass] = np.nan
                continue
            pred1 = preds1[idx]
            pred2 = preds2[idx]
            intersection = np.sum(pred1 == pred2)
            dice = (2. * intersection) / (len(pred1) + len(pred2) + 1e-8)
            dice_scores[currentClass] = dice

        if save_path:
            import matplotlib.pyplot as plt
            import os

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            class_labels = list(dice_scores.keys())
            scores = [dice_scores[k] for k in class_labels]
            
            # Crea figura e assi
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(class_labels, scores, color='skyblue')
            
            # Aggiungi valore sopra ogni barra
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
            
            ax.set_xlabel("Classe")
            ax.set_ylabel("DICE Score")
            ax.set_title(f"DICE Score tra {self.model1.__class__.__name__} e {self.model2.__class__.__name__}")
            ax.set_xticks(class_labels)
            ax.set_ylim(0, 1)  # DICE score è sempre tra 0 e 1
            
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"DICE score plot salvato in: {save_path}")
            plt.close()

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