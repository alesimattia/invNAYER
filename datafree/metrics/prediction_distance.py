def prediction_distance(teacher, student, dataset_root='./CIFAR10', batch_size=512, num_workers=4):
    """
    Calcola la distanza media tra le predizioni del modello teacher e quelle del modello studente per ogni classe.
    """

    import torch
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    import numpy as np
    from sklearn.metrics import pairwise_distances

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = teacher.to(device).eval()
    student = student.to(device).eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # Normalizzazione standard CIFAR-10
    ])

    '''
        datasets => torchvision.datasets
        Da documentazione Pytorch:
        train=False If True, creates dataset from training set, otherwise creates from test set.
    '''
    cifar_test = datasets.CIFAR10(root=dataset_root, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(cifar_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Distanze per classe
    class_distances = {i: [] for i in range(10)}

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            teacher_outputs = teacher(images)
            student_outputs = student(images)

            # Calcolo della distanza per ogni immagine
            distances = torch.norm(teacher_outputs - student_outputs, dim=1).cpu().numpy()

            # Raggruppa le distanze per classe
            for i, target in enumerate(targets.numpy()):
                class_distances[target].append(distances[i])

    # Calcolo della distanza media per ogni classe
    mean_distances = {class_idx: np.mean(class_distances[class_idx]) for class_idx in class_distances}

    return mean_distances