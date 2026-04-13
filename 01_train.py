import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import timm

# Configuration GPU et AMP
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Utilisation de l'appareil : {device}")

# Paramètres
BATCH_SIZE = 16
EPOCHS = 2
NUM_CLASSES = 3  # Car, Pedestrian, Truck
PATIENCE = 1
torch.backends.cudnn.benchmark = True # Optimisation CuDNN

def prepare_data():
    # Pipeline de données proxy (EuroSAT est un classique imagerie aérienne)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    
    print("Chargement du Proxy visuel (VisDrone Cropped)...")
    PROXY_DIR = './data/VisDrone_Proxy'
    if not os.path.exists(PROXY_DIR):
        raise Exception("Le dataset Proxy n'a pas été généré. Veuillez lancer `python 00_prepare_proxy_dataset.py` en premier.")
        
    dataset = datasets.ImageFolder(root=PROXY_DIR, transform=transform)
    
    # Division train/val (80% / 20%)
    dataset_size = len(dataset)
    indices = torch.randperm(dataset_size, generator=torch.Generator().manual_seed(42)).tolist()
    train_split = int(0.8 * dataset_size)
    
    train_subset = Subset(dataset, indices[:train_split])
    val_subset = Subset(dataset, indices[train_split:])
    
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=0)
    
    return train_loader, val_loader

def measure_inference_time(model, val_loader):
    model.eval()
    times = []
    with torch.no_grad():
        for i, (imgs, _) in enumerate(val_loader):
            imgs = imgs.to(device)
            # Warmup
            if i == 0:
                for _ in range(5): model(imgs)
            
            if device.type == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                
                model(imgs)
                
                end_event.record()
                torch.cuda.synchronize()
                times.append(start_event.elapsed_time(end_event) / imgs.size(0)) # ms par image
            else:
                start_t = time.time()
                model(imgs)
                end_t = time.time()
                times.append((end_t - start_t) * 1000 / imgs.size(0)) # ms par image
            
            if i >= 10: # Mesure sur 10 batches
                break
    
    avg_inf_time = sum(times) / len(times)
    print(f"Temps d'inférence moyen / image : {avg_inf_time:.2f} ms")
    return avg_inf_time

def train_model(model, name, train_loader, val_loader):
    print(f"\n--- Entraînement du modèle : {name} ---")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4) # Transfert learning
    scaler = torch.amp.GradScaler(device='cuda')
    
    best_loss = float('inf')
    early_stop_counter = 0
    
    # On gèle les premières couches pour accélérer (Transfer Learning partiel)
    # Pour la preuve de concept, on affine tout s'il y a le temps, ou on peut geler p. ex.
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            if device.type == 'cuda':
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
            running_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                if device.type == 'cuda':
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        outputs = model(imgs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * correct / total
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} - Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Early Stopping
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            early_stop_counter = 0
            os.makedirs('./models', exist_ok=True)
            torch.save(model.state_dict(), f'./models/{name}.pth')
            print("  --> Modèle sauvegardé !")
        else:
            early_stop_counter += 1
            if early_stop_counter >= PATIENCE:
                print("Early stopping triggerré.")
                break

    # Re-charger le meilleur modèle pour tester l'inférence
    model.load_state_dict(torch.load(f'./models/{name}.pth'))
    measure_inference_time(model, val_loader)

if __name__ == "__main__":
    train_loader, val_loader = prepare_data()
    
    # Modèle A : ResNet-50 (Baseline CNN) - Ré-entraînement sur VisDrone (3 classes)
    print("Initialisation ResNet-50...")
    resnet = timm.create_model('resnet50', pretrained=True, num_classes=NUM_CLASSES)
    train_model(resnet, 'Model_A_ResNet50', train_loader, val_loader)
    # L'architecture ViT Tiny de DeiT est très performante et légère
    print("Initialisation DeiT-Tiny (Transformer)...")
    deit = timm.create_model('deit_tiny_patch16_224', pretrained=True, num_classes=NUM_CLASSES)
    train_model(deit, 'Model_B_DeiT', train_loader, val_loader)
