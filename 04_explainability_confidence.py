import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import timm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ProjectedGradientDescent
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 3

def load_models():
    # ResNet50
    resnet = timm.create_model('resnet50', pretrained=False, num_classes=NUM_CLASSES).to(device)
    resnet.load_state_dict(torch.load('./models/Model_A_ResNet50.pth', map_location=device))
    resnet.eval()
    
    # DeiT
    deit = timm.create_model('deit_tiny_patch16_224', pretrained=False, num_classes=NUM_CLASSES).to(device)
    deit.load_state_dict(torch.load('./models/Model_B_DeiT.pth', map_location=device))
    deit.eval()
    
    return resnet, deit

def get_art_pgd(model):
    criterion = nn.CrossEntropyLoss()
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1).astype(np.float32)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1).astype(np.float32)
    classifier = PyTorchClassifier(model=model, clip_values=(0.0, 1.0), loss=criterion,
                                   optimizer=None, input_shape=(3, 224, 224), nb_classes=NUM_CLASSES,
                                   preprocessing=(mean, std), device_type='gpu' if torch.cuda.is_available() else 'cpu')
    return ProjectedGradientDescent(estimator=classifier, eps=0.05, eps_step=0.01, max_iter=10)

def generate_xai():
    print("=== Phase 5 : Expliquabilité Visuelle (Grad-CAM) ===")
    resnet, deit = load_models()
    
    # Charger une image spécifique (ex: Car de VisDrone)
    proxy_dir = './data/VisDrone_Proxy'
    car_dir = os.path.join(proxy_dir, 'Car')
    if not os.path.exists(car_dir) or len(os.listdir(car_dir)) == 0:
        print("Dataset absent.")
        return
        
    img_name = os.listdir(car_dir)[0]
    img_path = os.path.join(car_dir, img_name)
    
    rgb_img = (np.array(Image.open(img_path).resize((224, 224))) / 255.0).astype(np.float32)
    input_tensor = transforms.ToTensor()(rgb_img).unsqueeze(0).to(device)
    input_tensor = input_tensor.type(torch.float32)
    
    # Générer attaque vis-à-vis du ResNet
    pgd = get_art_pgd(resnet)
    input_adv_npy = pgd.generate(x=input_tensor.cpu().numpy())
    input_adv_tensor = torch.tensor(input_adv_npy, dtype=torch.float32).to(device)
    rgb_adv_img = np.transpose(input_adv_npy[0], (1, 2, 0))
    
    # Grad-CAM sur ResNet50 (layer4[-1] correspond à Bottleneck final dans torchvision/timm)
    target_layers = [resnet.layer4[-1]]
    
    with GradCAM(model=resnet, target_layers=target_layers) as cam:
        # Cam clean
        grayscale_cam_clean = cam(input_tensor=input_tensor, targets=None)[0, :]
        cam_image_clean = show_cam_on_image(rgb_img, grayscale_cam_clean, use_rgb=True)
        
        # Cam attack
        grayscale_cam_adv = cam(input_tensor=input_adv_tensor, targets=None)[0, :]
        cam_image_adv = show_cam_on_image(rgb_adv_img, grayscale_cam_adv, use_rgb=True)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(cam_image_clean)
    axes[0].set_title("ResNet-50 Attention (Sain)")
    axes[0].axis('off')
    
    axes[1].imshow(cam_image_adv)
    axes[1].set_title("ResNet-50 Attention (Attaque PGD)")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('xai_gradcam.png')
    print("-> XAI Heatmap générée : xai_gradcam.png")

def calculate_confidence():
    print("=== Phase 6 : Calibration de Confiance Softmax ===")
    resnet, deit = load_models()
    
    proxy_dir = './data/VisDrone_Proxy'
    dataset = datasets.ImageFolder(root=proxy_dir, transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]))
    
    # On prend une image pour analyser l'évolution du Softmax selon Epsilon
    img, label = dataset[50] # Un echantillon au pif
    x = img.unsqueeze(0).cpu().numpy()
    x_tensor = img.unsqueeze(0).to(device)
    
    # On définit classifier pour Deit
    art_deit = PyTorchClassifier(model=deit, clip_values=(0, 1), loss=nn.CrossEntropyLoss(),
                                input_shape=(3, 224, 224), nb_classes=NUM_CLASSES,
                                preprocessing=(np.array([0.485, 0.456, 0.406]).reshape(3,1,1),
                                               np.array([0.229, 0.224, 0.225]).reshape(3,1,1)))
    
    epsilons = [0.0, 0.02, 0.05, 0.1]
    conf_deit = []
    
    for eps in epsilons:
        if eps == 0.0:
            logits = deit((x_tensor - torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)) / torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device))
            conf = F.softmax(logits, dim=1).max().item() * 100
            conf_deit.append(conf)
        else:
            pgd = ProjectedGradientDescent(estimator=art_deit, eps=eps, eps_step=eps/4, max_iter=10)
            x_adv = torch.tensor(pgd.generate(x=x)).to(device)
            logits = deit((x_adv - torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)) / torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device))
            conf = F.softmax(logits, dim=1).max().item() * 100
            conf_deit.append(conf)
            
    plt.figure(figsize=(6, 4))
    plt.plot(epsilons, conf_deit, marker='o', color='green', linewidth=2)
    plt.title("Érosion de Confiance (Modèle Hybride/DeiT)")
    plt.xlabel("Force de l'attaque PGD (Epsilon)")
    plt.ylabel("Confiance de prédiction (Softmax %)")
    plt.grid()
    plt.tight_layout()
    plt.savefig('confidence_erosion.png')
    print("-> Graphique de confiance généré : confidence_erosion.png")

if __name__ == "__main__":
    generate_xai()
    calculate_confidence()
