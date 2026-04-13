import os
import urllib.request
import zipfile
from PIL import Image

# Configuration
DATA_DIR = "./data"
VISDRONE_VAL_URL = "https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-val.zip"
VISDRONE_DIR = os.path.join(DATA_DIR, "VisDrone")
PROXY_DIR = os.path.join(DATA_DIR, "VisDrone_Proxy")

# VisDrone categories to extract
# VisDrone classes: 1: pedestrian, 4: car, 6: truck
TARGET_CLASSES = {
    '1': 'Pedestrian',
    '4': 'Car',
    '6': 'Truck'
}
MAX_PER_CLASS = 200 # Limite pour la preuve de concept rapide sur CPU/VRAM modérée

def download_and_extract():
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "VisDrone2019-DET-val.zip")
    
    if not os.path.exists(zip_path):
        print(f"Téléchargement du Validation Set VisDrone (~100 Mo) depuis : {VISDRONE_VAL_URL}")
        urllib.request.urlretrieve(VISDRONE_VAL_URL, zip_path)
        print("Téléchargement terminé.")
        
    print("Extraction...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
        
    # L'extraction crée probablement un dossier "VisDrone2019-DET-val" ou "VisDrone"
    return os.path.join(DATA_DIR, "VisDrone2019-DET-val")

def create_proxy_dataset(base_dir):
    images_dir = os.path.join(base_dir, "images")
    annotations_dir = os.path.join(base_dir, "annotations")
    
    # Création de l'arborescence
    for class_name in TARGET_CLASSES.values():
        os.makedirs(os.path.join(PROXY_DIR, class_name), exist_ok=True)
        
    class_counts = {name: 0 for name in TARGET_CLASSES.values()}
    
    print("Démarrage du processus de 'Target Cropping' SAA...")
    
    total_images = len(os.listdir(images_dir))
    
    for idx, img_file in enumerate(os.listdir(images_dir)):
        if all(count >= MAX_PER_CLASS for count in class_counts.values()):
            print("\nQuotas atteints pour la preuve de concept ! Fin de l'extraction.")
            break
            
        if not img_file.endswith(('.jpg', '.png')):
            continue
            
        base_name = os.path.splitext(img_file)[0]
        ann_file = os.path.join(annotations_dir, base_name + ".txt")
        
        if not os.path.exists(ann_file):
            continue
            
        # Ouverture image
        img_path = os.path.join(images_dir, img_file)
        try:
            img = Image.open(img_path)
            
            with open(ann_file, 'r') as f:
                lines = f.readlines()
                for line_idx, line in enumerate(lines):
                    parts = line.strip().split(',')
                    if len(parts) < 6:
                        continue
                        
                    bbox_left, bbox_top, bbox_width, bbox_height, score, category = parts[:6]
                    if category in TARGET_CLASSES:
                        class_name = TARGET_CLASSES[category]
                        if class_counts[class_name] >= MAX_PER_CLASS:
                            continue
                            
                        left = int(bbox_left)
                        top = int(bbox_top)
                        right = left + int(bbox_width)
                        bottom = top + int(bbox_height)
                        
                        # Exclusion des boîtes absurdes ou corrompues
                        if right <= left or bottom <= top or int(bbox_width) < 10 or int(bbox_height) < 10:
                            continue
                            
                        # Crop
                        cropped_img = img.crop((left, top, right, bottom))
                        # Redimensionnement imposé par les modèles standards PyTorch
                        cropped_img = cropped_img.resize((224, 224), Image.Resampling.LANCZOS)
                        
                        save_path = os.path.join(PROXY_DIR, class_name, f"{base_name}_{line_idx}.jpg")
                        cropped_img.save(save_path)
                        class_counts[class_name] += 1
                        
        except Exception as e:
            print(f"Erreur sur {img_file}: {e}")
            
        if idx % 10 == 0:
            print(f"Progression... [{idx}/{total_images}] | Counts : {class_counts}")

    print("====================================")
    print("Dataset Proxy d'Imagerie Aérienne prêt !")
    print(f"Stocké dans : {PROXY_DIR}")
    for class_name, count in class_counts.items():
        print(f" - {class_name} : {count} patchs")

if __name__ == "__main__":
    extracted_dir = download_and_extract()
    create_proxy_dataset(extracted_dir)
