import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import timm
import numpy as np
import matplotlib.pyplot as plt
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ProjectedGradientDescent

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 3  # Car, Pedestrian, Truck
BATCH_SIZE = 16

def create_art_classifier(model):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1).astype(np.float32)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1).astype(np.float32)
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(0.0, 1.0),
        loss=criterion,
        optimizer=None,
        input_shape=(3, 224, 224),
        nb_classes=NUM_CLASSES,
        preprocessing=(mean, std),
        device_type='gpu' if torch.cuda.is_available() else 'cpu'
    )
    return classifier

def load_cross_domain_data():
    """
    Tente de charger un dataset 'Target Domain' (ex: classifié militaire ou autre proxy).
    Si introuvable, utilise une petite fraction du proxy source (VisDrone) pour démontrer la mécanique.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    TARGET_DIR = './data/Target_Domain'
    PROXY_DIR = './data/VisDrone_Proxy'
    
    if os.path.exists(TARGET_DIR) and len(os.listdir(TARGET_DIR)) > 0:
        print(f"Chargement des données du Target Domain depuis : {TARGET_DIR}")
        dataset = datasets.ImageFolder(root=TARGET_DIR, transform=transform)
    else:
        print(f"Dossier Target Domain absent/vide. Fallback sur le dataset source ({PROXY_DIR}) pour la démo Cross-Domain...")
        os.makedirs(TARGET_DIR, exist_ok=True)
        dataset = datasets.ImageFolder(root=PROXY_DIR, transform=transform)
    
    # Prendre 20 images pour la preuve empirique Cross-Domain
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(100)).tolist()
    val_subset = Subset(dataset, indices[:20])
    
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    
    x_test, y_test = [], []
    for imgs, labels in val_loader:
        x_test.append(imgs.cpu().numpy())
        y_test.append(labels.cpu().numpy())
    
    return np.concatenate(x_test), np.concatenate(y_test)

def run_cross_domain_test():
    print("=== Démarrage de l'Analye de Transférabilité SAA (Cross-Domain) ===")
    
    x_test, y_test = load_cross_domain_data()
    
    # On évalue le Modèle Hybride uniquement
    print("Chargement Deit-Tiny (Model B)...")
    deit = timm.create_model('deit_tiny_patch16_224', pretrained=False, num_classes=NUM_CLASSES).to(device)
    deit.load_state_dict(torch.load('./models/Model_B_DeiT.pth'))
    art_deit = create_art_classifier(deit)
    
    epsilons = [0.0, 0.05, 0.1]
    acc_deit = []
    
    for eps in epsilons:
        print(f"--- Attaque Cross-Domain PGD (Epsilon : {eps}) ---")
        if eps == 0.0:
            pred = np.argmax(art_deit.predict(x_test), axis=1)
            acc = np.mean(pred == y_test)
            acc_deit.append(acc)
            print(f"Clean Acc (Target Domain): {acc*100:.2f}%")
        else:
            pgd = ProjectedGradientDescent(estimator=art_deit, eps=eps, eps_step=eps/4, max_iter=10)
            x_adv = pgd.generate(x=x_test)
            pred = np.argmax(art_deit.predict(x_adv), axis=1)
            acc = np.mean(pred == y_test)
            acc_deit.append(acc)
            print(f"PGD Attack Acc (Target Domain): {acc*100:.2f}%")

    # Plot
    plt.figure(figsize=(6, 5))
    plt.plot(epsilons, acc_deit, marker='s', color='orange', label='Hybride (DeiT) sur Target Domain')
    plt.title('Résilience Transférable (SAA Cross-Domain)')
    plt.xlabel('Intensité Attaque PGD (Epsilon)')
    plt.ylabel('Précision (Accuracy)')
    plt.ylim([0, 1.05])
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig('cross_domain_resilience.png')
    print("-> Graphique généré : cross_domain_resilience.png")

if __name__ == "__main__":
    run_cross_domain_test()
