import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import timm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 3  # Car, Pedestrian, Truck
BATCH_SIZE = 16

def prepare_val_data():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # ART gère mieux si l'image est juste [0, 1]. Nous n'utilisons pas la normalisation ImageNet ici 
        # pour s'assurer que les perturbations d'ART sont bornées correctement [0, 1].
        # Si on normalise, il faut définir pré-process dans PyTorchClassifier.
    ])
    
    PROXY_DIR = './data/VisDrone_Proxy'
    dataset = datasets.ImageFolder(root=PROXY_DIR, transform=transform)
    
    # Prendre 20% pour l'intégration proxy / val
    dataset_size = len(dataset)
    indices = torch.randperm(dataset_size, generator=torch.Generator().manual_seed(42)).tolist()
    val_split = int(0.8 * dataset_size)
    
    # Preuve de concept ultra-rapide sur CPU : Max 20 images
    val_subset = Subset(dataset, indices[val_split:val_split+20])
    
    # Loader spécifique sans norrmalisation (on normalisera dans le modèle)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Convertir en NumPy pour ART
    x_test, y_test = [], []
    for imgs, labels in val_loader:
        x_test.append(imgs.cpu().numpy())
        y_test.append(labels.cpu().numpy())
    
    return np.concatenate(x_test), np.concatenate(y_test)

def create_art_classifier(model):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    # Nous ajoutons la normalisation ImageNet DANS le classifier ART
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

def eval_and_plot():
    x_test, y_test = prepare_val_data()
    
    # Charger Modèle A
    resnet = timm.create_model('resnet50', pretrained=False, num_classes=NUM_CLASSES).to(device)
    resnet.load_state_dict(torch.load('./models/Model_A_ResNet50.pth'))
    art_resnet = create_art_classifier(resnet)
    
    # Charger Modèle B
    deit = timm.create_model('deit_tiny_patch16_224', pretrained=False, num_classes=NUM_CLASSES).to(device)
    deit.load_state_dict(torch.load('./models/Model_B_DeiT.pth'))
    art_deit = create_art_classifier(deit)
    
    epsilons = [0.0, 0.01, 0.03, 0.05, 0.1]
    
    acc_resnet_fgsm, acc_deit_fgsm = [], []
    acc_resnet_pgd, acc_deit_pgd = [], []
    
    # Pour matrice de confusion (Max Epsilon = 0.05 PGD)
    y_pred_res_max, y_pred_deit_max = None, None
    y_pred_res_clean, y_pred_deit_clean = None, None
    
    print("Évaluation en cours...")
    for eps in epsilons:
        print(f"--- Epsilon : {eps} ---")
        if eps == 0.0:
            pred_res = np.argmax(art_resnet.predict(x_test), axis=1)
            pred_deit = np.argmax(art_deit.predict(x_test), axis=1)
            
            y_pred_res_clean = pred_res
            y_pred_deit_clean = pred_deit
            
            acc_res = np.mean(pred_res == y_test)
            acc_dt = np.mean(pred_deit == y_test)
            
            acc_resnet_fgsm.append(acc_res); acc_resnet_pgd.append(acc_res)
            acc_deit_fgsm.append(acc_dt); acc_deit_pgd.append(acc_dt)
            print(f"Clean Acc -> ResNet: {acc_res:.2f} | DeiT: {acc_dt:.2f}")
            continue
            
        # FGSM
        fgsm_res = FastGradientMethod(estimator=art_resnet, eps=eps)
        fgsm_deit = FastGradientMethod(estimator=art_deit, eps=eps)
        
        x_adv_res_fgsm = fgsm_res.generate(x=x_test)
        x_adv_deit_fgsm = fgsm_deit.generate(x=x_test)
        
        acc_res_f = np.mean(np.argmax(art_resnet.predict(x_adv_res_fgsm), axis=1) == y_test)
        acc_dt_f = np.mean(np.argmax(art_deit.predict(x_adv_deit_fgsm), axis=1) == y_test)
        acc_resnet_fgsm.append(acc_res_f)
        acc_deit_fgsm.append(acc_dt_f)
        
        # PGD
        pgd_res = ProjectedGradientDescent(estimator=art_resnet, eps=eps, eps_step=eps/4, max_iter=10)
        pgd_deit = ProjectedGradientDescent(estimator=art_deit, eps=eps, eps_step=eps/4, max_iter=10)
        
        x_adv_res_pgd = pgd_res.generate(x=x_test)
        x_adv_deit_pgd = pgd_deit.generate(x=x_test)
        
        pred_res_p = np.argmax(art_resnet.predict(x_adv_res_pgd), axis=1)
        pred_dt_p = np.argmax(art_deit.predict(x_adv_deit_pgd), axis=1)
        
        acc_res_p = np.mean(pred_res_p == y_test)
        acc_dt_p = np.mean(pred_dt_p == y_test)
        acc_resnet_pgd.append(acc_res_p)
        acc_deit_pgd.append(acc_dt_p)
        
        if eps == 0.05:
            y_pred_res_max = pred_res_p
            y_pred_deit_max = pred_dt_p

        print(f"FGSM Acc -> ResNet: {acc_res_f:.2f} | DeiT: {acc_dt_f:.2f}")
        print(f"PGD  Acc -> ResNet: {acc_res_p:.2f} | DeiT: {acc_dt_p:.2f}")

    # Plot des courbes de résilience
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epsilons, acc_resnet_fgsm, marker='o', label='Modèle A (ResNet) - FGSM')
    plt.plot(epsilons, acc_deit_fgsm, marker='s', label='Modèle B (DeiT / Hybride) - FGSM')
    plt.title('Résilience face à l\'attaque FGSM')
    plt.xlabel('Budget d\'attaque Epsilon')
    plt.ylabel('Précision (Accuracy)')
    plt.ylim([0, 1.05])
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(epsilons, acc_resnet_pgd, marker='o', label='Modèle A (ResNet) - PGD')
    plt.plot(epsilons, acc_deit_pgd, marker='s', label='Modèle B (DeiT / Hybride) - PGD')
    plt.title('Résilience face à l\'attaque PGD')
    plt.xlabel('Budget d\'attaque Epsilon')
    plt.ylabel('Précision (Accuracy)')
    plt.ylim([0, 1.05])
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig('resilience_curves.png')
    print("Graphique de résilience sauvegardé : resilience_curves.png")

    # Matrices de confusion
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    def plot_cm(y_true, y_pred, ax, title):
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(NUM_CLASSES))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
        ax.set_title(title)
        ax.set_ylabel('Vérité Terrain')
        ax.set_xlabel('Prédiction')

    plot_cm(y_test, y_pred_res_clean, axes[0, 0], 'Modèle A (Sain)')
    plot_cm(y_test, y_pred_deit_clean, axes[0, 1], 'Modèle B (Sain)')
    
    # Epsilon = 0.05
    plot_cm(y_test, y_pred_res_max, axes[1, 0], 'Modèle A (PGD eps=0.05)')
    plot_cm(y_test, y_pred_deit_max, axes[1, 1], 'Modèle B (PGD eps=0.05)')

    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    print("Matrices de confusion sauvegardées : confusion_matrices.png")
    
    # Vérification Critères
    gap_pgd = (acc_deit_pgd[-1] - acc_resnet_pgd[-1]) * 100
    print(f"\n--- Validation des Critères ---")
    print(f"Accuracy Drop PGD(max_eps) - ResNet: {(acc_resnet_pgd[0] - acc_resnet_pgd[-1])*100:.2f}%")
    print(f"Accuracy Drop PGD(max_eps) - DeiT: {(acc_deit_pgd[0] - acc_deit_pgd[-1])*100:.2f}%")
    print(f"Écart de robustesse (B vs A) sous PGD max: {gap_pgd:.2f}%")
    if gap_pgd >= 10:
        print("✓ SUCCESS: Avantage supérieur à 10% pour le modèle hybride validé.")
    else:
        print("x ECHEC: Écart inférieur à 10%.")

if __name__ == "__main__":
    eval_and_plot()
