import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd
import numpy as np
from PIL import Image
import os
import time
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, accuracy_score, confusion_matrix
import copy # Para guardar el mejor modelo
import torch.nn.functional as F # Necesario para F.log_softmax, F.softmax

from tqdm import tqdm



# python3 train_chexpert_kd_20.py --data_dir /workspace/WORKS/DATA/CheXpert-v1.0-small     --img_dir /workspace/WORKS/DATA/     --output_dir chexpert_multistate_results_01052025_2     --epochs 1     --batch_size 1     --lr 0.0001     --img_size 224     --teacher_model densenet121     --student_model mobilenet_v2     --classes Atelectasis Cardiomegaly Consolidation Edema 'Pleural Effusion' > Salida_01052025_2.txt


print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- 1. Argument Parser ---
parser = argparse.ArgumentParser(description="Train Teacher/Student on CheXpert with 4-bit state labels")
parser.add_argument('--data_dir', type=str, required=True, help='Base directory containing CheXpert CSV files (train.csv, valid.csv, test.csv)')
parser.add_argument('--img_dir', type=str, required=True, help='Base directory containing CheXpert image folders (e.g., where CheXpert-v1.0 folder resides)')
parser.add_argument('--output_dir', type=str, default='./chexpert_output', help='Directory to save models, plots, and results')
parser.add_argument('--classes', nargs='+', default=['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion'], help='List of 5 pathology class names (columns in CSV)')
parser.add_argument('--img_size', type=int, default=224, help='Image size (input to model)')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and evaluation')
parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
parser.add_argument('--teacher_model', type=str, default='densenet121', help='Teacher model architecture (e.g., densenet121)')
parser.add_argument('--student_model', type=str, default='mobilenet_v2', help='Student model architecture (e.g., mobilenet_v2)')
parser.add_argument('--use_pretrained', action='store_true', help='Use ImageNet pretrained weights (affects first layer if modifying)')
parser.add_argument('--kd_alpha', type=float, default=0.3, help='Weight for standard loss in KD (alpha). Weight for KD loss is (1-alpha).')
parser.add_argument('--kd_temperature', type=float, default=4.0, help='Temperature for softening outputs in KD.')

args = parser.parse_args()

# --- Crear directorio de salida ---
os.makedirs(args.output_dir, exist_ok=True)

# --- Constantes ---
NUM_PATHOLOGIES = len(args.classes)
STATE_SUFFIXES = ['_Present', '_Absent', '_Uncertain', '_NotMentioned'] # Usados para nombrar columnas
NUM_STATES = len(STATE_SUFFIXES) # Present, Absent, Uncertain, Not Mentioned
NUM_OUTPUT_NEURONS = NUM_PATHOLOGIES * NUM_STATES # 5 * 4 = 20

if NUM_PATHOLOGIES != 5:
    raise ValueError("Este script está diseñado para exactamente 5 patologías.")

if NUM_OUTPUT_NEURONS != 20:
     raise ValueError("La configuración de patologías y estados no resulta en 20 neuronas de salida.")



# --- 2. Dataset Definition ---

class ChexpertDataset20BitLabels(Dataset):
    def __init__(self, csv_path, img_dir, pathologies, state_suffixes, transform=None):
        self.img_labels_df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.pathologies = pathologies
        # Generar los nombres de las 20 columnas de etiquetas
        self.label_columns = [f"{p}{s}" for p in pathologies for s in state_suffixes]
        # Verificar que todas las columnas existen
        if not all(col in self.img_labels_df.columns for col in self.label_columns):
             missing = [col for col in self.label_columns if col not in self.img_labels_df.columns]
             raise ValueError(f"El CSV {csv_path} no contiene las columnas de etiquetas de 20 bits: {missing}")


    def __len__(self):
        return len(self.img_labels_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.img_labels_df.iloc[idx]
        img_path_partial = row['Path']
        # img_path = os.path.join(self.img_dir, '/'.join(img_path_partial.split('/')[1:])) # ¡Ajustar si es necesario!
        img_path = os.path.join(self.img_dir, img_path_partial)

        try:
            image = Image.open(img_path).convert('L') # Cargar como 'L'
        except Exception as e:
            print(f"ADVERTENCIA: Error cargando {img_path}: {e}. Saltando índice {idx}.")
            return None # Manejar con collate_fn

        if self.transform:
            image = self.transform(image)
        else:
             image = transforms.ToTensor()(image)
             if image.shape[0] == 1: image = image.repeat(3, 1, 1) # Asegurar 3 canales si no hay transform

        # Obtener directamente las 20 etiquetas precalculadas
        label_values = row[self.label_columns].values.astype(np.float32)
        label_tensor = torch.from_numpy(label_values) # Shape: (20,)

        return image, label_tensor

# Al crear el Dataset en el main script, pasar los state_suffixes
# state_suffixes = ['_Present', '_Absent', '_Uncertain', '_NotMentioned']
# train_dataset = ChexpertDataset20BitLabels(train_csv_path, args.img_dir, args.classes, state_suffixes, transform=train_transform)
# ... etc ...
    

class ChexpertDatasetMultiState(Dataset):
    """
    Dataset CheXpert que devuelve etiquetas como índices de clase (0-3)
    para cada una de las 5 patologías.
    Mapping: Present=0, Absent=1, Uncertain=2, NotMentioned=3
    """
    def __init__(self, csv_path, img_dir, classes_list, transform=None, mode='train'):
        self.img_labels_df = pd.read_csv(csv_path)
        # Llenar NaNs en las columnas de etiquetas ANTES de mapear
        # Asumimos que NaNs en etiquetas significan "No Mencionado"
        self.img_labels_df[classes_list] = self.img_labels_df[classes_list].fillna(np.nan) # Asegurar que sean NaN reales
        self.img_dir = img_dir
        self.transform = transform
        self.classes = classes_list
        self.mode = mode # 'train', 'valid', 'test'

        # Mapping de valores crudos a índice de clase (0-3)
        self.state_mapping = {
            1.0: 0,  # Present
            0.0: 1,  # Absent
           -1.0: 2,  # Uncertain
            np.nan: 3 # Not Mentioned
        }

    def __len__(self):
        return len(self.img_labels_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.img_labels_df.iloc[idx]
        img_path_partial = row['Path']

        # Construir ruta de imagen (¡¡¡AJUSTAR SI ES NECESARIO!!!)
        # Asume que img_dir es la carpeta que contiene 'CheXpert-v1.0/'
        # y Path es como 'CheXpert-v1.0/train/patient...'
        img_path = os.path.join(self.img_dir, img_path_partial)

        try:
            # Cargar como 'L' y convertir a 3 canales en transformaciones
            image = Image.open(img_path).convert('L')
        except FileNotFoundError:
            print(f"ADVERTENCIA: Imagen no encontrada {img_path}. Saltando índice {idx}.")
            return None # Manejar con collate_fn
        except Exception as e:
            print(f"ADVERTENCIA: Error cargando {img_path}: {e}. Saltando índice {idx}.")
            return None

        # Aplicar transformaciones
        if self.transform:
            image = self.transform(image)
        else: # Asegurar que sea un tensor si no hay transformaciones
             image = transforms.ToTensor()(image)
             if image.shape[0] == 1: # Replicar si ToTensor no creó 3 canales
                 image = image.repeat(3, 1, 1)

        # Codificar etiquetas a índices de clase (0-3)
        label_indices_list = []
        raw_labels = row[self.classes].values

        for raw_label in raw_labels:
            mapped_index = self.state_mapping.get(raw_label, self.state_mapping[np.nan]) # Default a NotMentioned si hay valor raro
            if pd.isna(raw_label): # Manejo explícito de NaN para asegurar mapeo correcto
                 mapped_index = self.state_mapping[np.nan]
            label_indices_list.append(mapped_index)

        # Convertir a LongTensor (requerido por CrossEntropyLoss)
        label_indices_tensor = torch.tensor(label_indices_list, dtype=torch.long) # Shape: (5,)

        # Retornar solo imagen y tensor de índices
        return image, label_indices_tensor

# --- Collate function para manejar errores de carga ---
def collate_fn_skip_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        # Devuelve None si todo el lote falló, necesita manejo especial en el bucle
        # O devuelve tensores vacíos con forma correcta si prefieres
        print("ADVERTENCIA: Lote completo falló al cargar, devolviendo None.")
        return None, None
    return torch.utils.data.dataloader.default_collate(batch)

# --- 3. Transformaciones ---
# Usar normalización de ImageNet es común, incluso si no se usan pesos preentrenados
# Asegura que la entrada tenga 3 canales para modelos estándar
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    # Augmentations (opcional pero recomendado)
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1), # Ligero ajuste
    transforms.Grayscale(num_output_channels=3), # Asegura 3 canales
    transforms.ToTensor(),
    normalize,
])

val_test_transform = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.Grayscale(num_output_channels=3), # Asegura 3 canales
    transforms.ToTensor(),
    normalize,
])

# --- 4. Crear Datasets y DataLoaders ---
try:

    state_suffixes = ['_Present', '_Absent', '_Uncertain', '_NotMentioned']
# train_dataset = ChexpertDataset20BitLabels(train_csv_path, args.img_dir, args.classes, state_suffixes, transform=train_transform)
#
        #train_stratified_split_20bit
    train_dataset = ChexpertDataset20BitLabels(os.path.join(args.data_dir, 'train_stratified_split_20bit.csv'), args.img_dir, args.classes, STATE_SUFFIXES, transform=train_transform)
    valid_dataset = ChexpertDataset20BitLabels(os.path.join(args.data_dir, 'valid_20bit.csv'), args.img_dir, args.classes, STATE_SUFFIXES, transform=val_test_transform)
    test_dataset = ChexpertDataset20BitLabels(os.path.join(args.data_dir, 'test_stratified_split_20bit.csv'), args.img_dir, args.classes, STATE_SUFFIXES, transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_skip_none)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_skip_none)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_skip_none)

    print("\n<<< Data Information >>>")
    print(f"Train data : {len(train_dataset)}")
    print(f"Valid data : {len(valid_dataset)}")
    print(f"Test data : {len(test_dataset)}")
    if len(train_dataset) == 0 or len(valid_dataset) == 0 or len(test_dataset) == 0:
         raise ValueError("Uno o más datasets están vacíos. Verifica las rutas de los CSV.")

except FileNotFoundError as e:
    print(f"\nERROR: No se encontró el archivo CSV: {e}")
    print("Por favor, asegúrate que 'train.csv', 'valid.csv', 'test.csv' existen en --data_dir")
    exit()
except ValueError as e:
     print(f"\nERROR: {e}")
     exit()

# --- 5. Model Definition ---
def create_model(model_name, num_classes_out, use_pretrained=False):
    """Crea el modelo (DenseNet o MobileNet) y ajusta la capa final."""
    model = None
    weights = None
    if use_pretrained:
        print(f"Attempting to load {model_name} with pretrained weights.")
        if model_name == 'densenet121':
            weights = models.DenseNet121_Weights.IMAGENET1K_V1
        elif model_name == 'mobilenet_v2':
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
        # Añadir más modelos/pesos aquí si es necesario
        else:
             print(f"Advertencia: Pesos preentrenados no definidos para {model_name}. Cargando sin preentrenar.")

    print(f"Loading {model_name} architecture...")
    if model_name == 'densenet121':
        model = models.densenet121(weights=weights)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes_out)
        print(f"Replaced DenseNet classifier, input features: {num_ftrs}, output features: {num_classes_out}")
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=weights)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Linear(num_ftrs, num_classes_out)
        print(f"Replaced MobileNetV2 classifier, input features: {num_ftrs}, output features: {num_classes_out}")
    # Añadir más arquitecturas aquí si se desea
    else:
        raise ValueError(f"Model architecture '{model_name}' not supported.")

    # No modificamos la primera capa porque las transformaciones aseguran 3 canales de entrada

    return model


# --- 6. Loss Functions ---
criterion_bce = nn.BCEWithLogitsLoss() # Para hard loss (train y val)
criterion_bce_elementwise = nn.BCEWithLogitsLoss(reduction='none') # Para calcular loss por patología
criterion_mse = nn.MSELoss() # Para soft loss (KD) entre logits

criterion_elementwise = nn.BCEWithLogitsLoss(reduction='none')

# --- 7. Training and Validation Functions ---
def train_epoch(model_name_tag, model, dataloader, criterion, optimizer, device, num_pathologies, pathology_names, current_epoch, total_epochs):
    model.train()
    total_loss_accum = 0.0
    pathology_losses_accum = torch.zeros(num_pathologies, device=device) # Acumulador por patología
    num_samples_processed = 0 # Contador de muestras en lugar de batches

      # Envolver dataloader con tqdm
    

    pbar = tqdm(dataloader, desc='{} Step at start {}; Training epoch {}/{}'.format(model_name_tag, num_samples_processed, current_epoch+1, total_epochs), leave=False)      
    
    for images, target_20bit in pbar:
        # Manejar lotes vacíos si collate_fn devolvió None
        if images is None or target_20bit is None:
            print("Saltando lote vacío en entrenamiento.")
            continue

        images, target_20bit = images.to(device), target_20bit.to(device)
        batch_size = images.size(0)
        optimizer.zero_grad()

        # Forward pass -> Salida de 20 logits
        outputs = model(images) # Shape: (batch_size, 20)

        # Calcular loss total para backward
        batch_loss = criterion_bce(outputs, target_20bit)

        if num_samples_processed==0:
            print("images : ", images.shape)
            print("images : ", images)
            print("target_indices_5 : ", target_20bit.shape)
            print("target_indices_5 : ", target_20bit)
            print("outputs : ", outputs.shape)
            print("outputs : ", outputs)


        # Calcular loss por patología (para logging)
        with torch.no_grad(): # No necesitamos gradientes para esto
             loss_20_elements = criterion_elementwise(outputs, target_20bit) # (B, 20)
             loss_reshaped = loss_20_elements.view(batch_size, num_pathologies, NUM_STATES) # (B, 5, 4)
             loss_per_pathology_sample = loss_reshaped.mean(dim=2) # (B, 5)
             batch_pathology_losses = loss_per_pathology_sample.mean(dim=0) # (5,)


        # Backward pass y optimización
        batch_loss.backward()
        optimizer.step()

        # --- Acumular ---
        total_loss_accum += batch_loss.item() * batch_size # Usa la loss total promediada del batch
        pathology_losses_accum += batch_pathology_losses.detach() * batch_size # Acumula las promediadas por patología
        num_samples_processed += batch_size


        # Actualizar la descripción de la barra (opcional)
        pbar.set_postfix(loss=f"{batch_loss.item():.4f}")

    # --- Calcular Promedios de la Época ---
    avg_epoch_loss = total_loss_accum / num_samples_processed if num_samples_processed > 0 else 0
    avg_pathology_losses_epoch = pathology_losses_accum / num_samples_processed if num_samples_processed > 0 else torch.zeros(num_pathologies)

    # Devolver la loss total y las pérdidas por patología
    return avg_epoch_loss, avg_pathology_losses_epoch.cpu().numpy()


def validate_epoch(model_name_tag, model, dataloader, criterion_bce, criterion_elementwise, device, num_pathologies, pathology_names, current_epoch, total_epochs):
    model.eval()
    total_loss_accum = 0.0
    pathology_losses_accum = torch.zeros(num_pathologies, device=device)
    num_samples_processed = 0
    all_outputs_20 = []
    all_targets_20bit = []

    pbar = tqdm(dataloader, desc=f"{model_name_tag} Epoch {current_epoch+1}/{total_epochs} [Val]  ", leave=False)


    with torch.no_grad():
        for images, target_20bit in pbar:
            if images is None or target_20bit is None:
                print("Saltando lote vacío en validación.")
                continue

            images, target_20bit = images.to(device), target_20bit.to(device)
            batch_size = images.size(0)

            outputs = model(images) # Shape: (batch_size, 20)
            batch_loss = criterion_bce(outputs, target_20bit)

            # Calcular loss por patología (para logging)
            loss_20_elements = criterion_elementwise(outputs, target_20bit) # (B, 20)
            loss_reshaped = loss_20_elements.view(batch_size, num_pathologies, NUM_STATES) # (B, 5, 4)
            loss_per_pathology_sample = loss_reshaped.mean(dim=2) # (B, 5)
            batch_pathology_losses = loss_per_pathology_sample.mean(dim=0) # (5,)

            # Acumular
            total_loss_accum += batch_loss.item() * batch_size
            pathology_losses_accum += batch_pathology_losses.detach() * batch_size
            num_samples_processed += batch_size

            pbar.set_postfix(loss=f"{batch_loss.item():.4f}")
            all_outputs_20.append(outputs.cpu())
            all_targets_20bit.append(target_20bit.cpu())


    avg_epoch_loss = total_loss_accum / num_samples_processed if num_samples_processed > 0 else 0
    avg_pathology_losses_epoch = pathology_losses_accum / num_samples_processed if num_samples_processed > 0 else torch.zeros(num_pathologies)

    final_outputs = torch.cat(all_outputs_20, dim=0) if all_outputs_20 else torch.empty(0, NUM_OUTPUT_NEURONS)
    final_targets = torch.cat(all_targets_20bit, dim=0) if all_targets_20bit else torch.empty(0, NUM_OUTPUT_NEURONS)

    return avg_epoch_loss, avg_pathology_losses_epoch.cpu().numpy(), final_outputs, final_targets

# *** NUEVA FUNCIÓN DE ENTRENAMIENTO CON KD ***
def train_epoch_kd(student_model_name_tag, student_model, teacher_model, dataloader, criterion_hard_elementwise, criterion_soft_mse,
                   optimizer, device, num_pathologies, pathology_names,
                   current_epoch, total_epochs, alpha):
    student_model.train() # Poner student en modo train
    teacher_model.eval()  # Teacher siempre en modo eval

    total_loss_accum = 0.0
    hard_loss_accum = 0.0
    soft_loss_accum = 0.0
    pathology_losses_accum = torch.zeros(num_pathologies, device=device) # Basado en hard loss
    num_samples_processed = 0

    pbar = tqdm(dataloader, desc=f"{student_model_name_tag} Epoch {current_epoch+1}/{total_epochs} [KD Train]", leave=False)

    criterion_hard_mean = nn.BCEWithLogitsLoss() # Para calcular el hard loss promedio para alpha

    for images, target_20bit in pbar:
        if images is None or target_20bit is None:
            continue

        images, target_20bit = images.to(device), target_20bit.to(device)
        batch_size = images.size(0)
        optimizer.zero_grad()

        # --- Forward pass Teacher (sin gradientes) ---
        with torch.no_grad():
            teacher_outputs = teacher_model(images) # Shape: (batch_size, 20)
            
        # --- Forward pass Student ---
        student_outputs = student_model(images) # Shape: (batch_size, 20)

        # --- Calcular Hard Loss ---
        hard_loss = criterion_hard_mean(student_outputs, target_20bit)
        
        # Calcular hard loss por patología (para logging)
        with torch.no_grad():
            loss_20_elements = criterion_hard_elementwise(student_outputs, target_20bit) # (B, 20)
            loss_reshaped = loss_20_elements.view(batch_size, num_pathologies, NUM_STATES) # (B, 5, 4)
            loss_per_pathology_sample = loss_reshaped.mean(dim=2) # (B, 5)
            batch_pathology_losses = loss_per_pathology_sample.mean(dim=0) # (5,)


        # --- Calcular Soft Loss (MSE en logits) ---
        soft_loss = criterion_soft_mse(student_outputs, teacher_outputs.detach())

        # --- Combinar ---
        combined_loss = alpha * hard_loss + (1.0 - alpha) * soft_loss

        # --- Backward y Optimize ---
        combined_loss.backward()
        optimizer.step()

        # --- Acumular ---
        total_loss_accum += combined_loss.item() * batch_size
        hard_loss_accum += hard_loss.item() * batch_size
        soft_loss_accum += soft_loss.item() * batch_size
        pathology_losses_accum += batch_pathology_losses.detach() * batch_size
        num_samples_processed += batch_size
        pbar.set_postfix(TotalL=f"{combined_loss.item():.4f}", HardL=f"{hard_loss.item():.4f}", SoftL=f"{soft_loss.item():.4f}")

    # --- Promedios de Época ---
    avg_total = total_loss_accum / num_samples_processed if num_samples_processed > 0 else 0
    avg_hard = hard_loss_accum / num_samples_processed if num_samples_processed > 0 else 0
    avg_soft = soft_loss_accum / num_samples_processed if num_samples_processed > 0 else 0
    avg_pathology_losses_epoch = pathology_losses_accum / num_samples_processed if num_samples_processed > 0 else torch.zeros(num_pathologies)

    # Devolver pérdidas relevantes
    return avg_total, avg_pathology_losses_epoch.cpu().numpy(), avg_hard, avg_soft

# --- 8. Main Training Execution Function ---
def run_training(model_name_tag, model, train_loader, valid_loader, criterion_bce, criterion_elementwise, optimizer, epochs, device, num_pathologies, pathology_names, output_dir):
    print(f"\n--- Training {model_name_tag} ---")
    history = {
        'train_loss': [], 'val_loss': [],
        'train_loss_per_pathology': [[] for _ in range(num_pathologies)],
        'val_loss_per_pathology': [[] for _ in range(num_pathologies)]
    }
    best_val_loss = float('inf')
    best_model_wts = None
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()

        # Training
        train_loss, train_path_losses = train_epoch(model_name_tag, model, train_loader, criterion_elementwise, optimizer, device, num_pathologies, pathology_names, epoch, epochs)
        history['train_loss'].append(train_loss)
        for i in range(num_pathologies):
            history['train_loss_per_pathology'][i].append(train_path_losses[i])

        # Validation
        val_loss, val_path_losses, _, _ = validate_epoch(model_name_tag, model, valid_loader, criterion_bce, criterion_elementwise, device, num_pathologies, pathology_names, epoch, epochs)
        history['val_loss'].append(val_loss)
        for i in range(num_pathologies):
            history['val_loss_per_pathology'][i].append(val_path_losses[i])

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_time:.2f}s")
        for i, name in enumerate(pathology_names):
             print(f"  {name}: Train Loss: {train_path_losses[i]:.4f}, Val Loss: {val_path_losses[i]:.4f}")


        # Guardar el mejor modelo basado en la pérdida de validación total
        if val_loss < best_val_loss:
            print(f"  Validation loss improved ({best_val_loss:.4f} -> {val_loss:.4f}). Saving model...")
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            model_save_path = os.path.join(output_dir, f'{model_name_tag}_best_model.pth')
            torch.save(best_model_wts, model_save_path)

    total_training_time = time.time() - start_time
    print(f"--- Training Finished for {model_name_tag} ---")
    print(f"Total Training Time: {total_training_time // 60:.0f}m {total_training_time % 60:.0f}s")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    if best_model_wts:
         print(f"Best model saved to {model_save_path}")
         # Cargar el mejor modelo para evaluación final
         model.load_state_dict(best_model_wts)
    else:
         print("Advertencia: No se guardó ningún modelo (posiblemente no hubo mejora o error).")


    return model, history

# --- 8. Main Training Execution Function (MODIFICADA para KD) ---
# Renombrar la función original a run_training_standard si quieres mantenerla
# Esta nueva función maneja el flujo de KD
def run_training_kd(student_model_name_tag, student_model, teacher_model, # Pasar teacher
                    train_loader, valid_loader,
                    criterion_hard_elementwise, criterion_soft_mse, # Pasar ambas losses
                    optimizer, epochs, device, num_pathologies, pathology_names,
                    output_dir, alpha, temperature): # Pasar alpha y T

    print(f"\n--- Training {student_model_name_tag} with Knowledge Distillation ---")
    history = {
        'train_loss': [], 'val_loss': [], 'train_hard_loss': [], 'train_soft_loss': [], # Añadir histórico de componentes KD
        'train_loss_per_pathology': [[] for _ in range(num_pathologies)],
        'val_loss_per_pathology': [[] for _ in range(num_pathologies)] # Val loss sigue siendo hard loss
    }
    best_val_loss = float('inf')
    best_model_wts = None
    start_time = time.time()

    # Asegurarse que el teacher esté en modo evaluación
    teacher_model.eval()

    criterion_hard_mean = nn.BCEWithLogitsLoss() # Para la parte 'hard' del alpha

    for epoch in range(epochs):
        epoch_start_time = time.time()

        # Training con KD
        train_total_loss, train_path_losses, train_hard_loss, train_soft_loss = train_epoch_kd(student_model_name_tag,
            student_model, teacher_model, train_loader, criterion_hard_elementwise, criterion_soft_mse, optimizer, device,
            num_pathologies, pathology_names, epoch, epochs, alpha
        )
        history['train_loss'].append(train_total_loss)
        history['train_hard_loss'].append(train_hard_loss)
        history['train_soft_loss'].append(train_soft_loss)
        for i in range(num_pathologies):
            history['train_loss_per_pathology'][i].append(train_path_losses[i]) # Sigue siendo basado en hard loss

        # Validation (usa la validación estándar, evaluando solo con hard loss)
        # Asumiendo que validate_epoch fue renombrada o es la original:
        val_loss, val_path_losses, val_outputs, val_targets = validate_epoch(student_model_name_tag, # O validate_epoch_standard
            student_model, valid_loader, criterion_hard_mean, criterion_hard_elementwise, device, num_pathologies, pathology_names, epoch, epochs
        )
        history['val_loss'].append(val_loss)
        for i in range(num_pathologies):
            history['val_loss_per_pathology'][i].append(val_path_losses[i])

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{epochs} | Train Loss (Total): {train_total_loss:.4f} (Hard: {train_hard_loss:.4f}, Soft: {train_soft_loss:.4f}) | Val Loss: {val_loss:.4f} | Time: {epoch_time:.2f}s")
        # Opcional: Imprimir pérdidas por patología si no es muy verboso

        # Guardar el mejor modelo student (KD) basado en la pérdida de validación (hard loss)
        if val_loss < best_val_loss:
            print(f"  Validation loss improved ({best_val_loss:.4f} -> {val_loss:.4f}). Saving KD student model...")
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(student_model.state_dict())
            model_save_path = os.path.join(output_dir, f'{student_model_name_tag}_KD_best_model.pth')
            torch.save(best_model_wts, model_save_path)

    total_training_time = time.time() - start_time
    print(f"--- KD Training Finished for {student_model_name_tag} ---")
    print(f"Total KD Training Time: {total_training_time // 60:.0f}m {total_training_time % 60:.0f}s")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    if best_model_wts:
         print(f"Best KD student model saved to {model_save_path}")
         student_model.load_state_dict(best_model_wts) # Cargar el mejor para evaluación
    else:
         print("Advertencia: No se guardó ningún modelo KD (posiblemente no hubo mejora o error).")

    return student_model, history


# --- 9. Plotting Functions ---
def plot_learning_curves(history, num_pathologies, pathology_names, output_dir, model_name_tag, include_kd_losses=False):
    if not history['train_loss']: # Verificar si el historial está vacío
         print(f"ADVERTENCIA: Historial vacío para {model_name_tag}. No se pueden generar curvas de aprendizaje.")
         return

    epochs_range = range(1, len(history['train_loss']) + 1)
    num_plots_pathology = len(history.get('train_loss_per_pathology', [[]]*num_pathologies)[0]) > 0

    # Calcular filas necesarias
    ncols = 2
    num_base_plots = 1 # Siempre plot de loss total
    if include_kd_losses and 'train_hard_loss' in history and history['train_hard_loss']:
         num_base_plots += 1 # Plot para componentes KD
    if num_plots_pathology:
         num_base_plots += num_pathologies

    num_rows = (num_base_plots + ncols - 1) // ncols

    plt.figure(figsize=(18, 6 * num_rows))
    plot_index = 1 # Índice actual del subplot

    # Plot Total Loss
    ax = plt.subplot(num_rows, ncols, plot_index)
    ax.plot(epochs_range, history['train_loss'], 'bo-', label='Total Training Loss')
    ax.plot(epochs_range, history['val_loss'], 'ro-', label='Validation Loss')
    ax.set_title(f'{model_name_tag} - Total Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    plot_index += 1

    # Plot Hard vs Soft Loss (si aplica)
    if include_kd_losses and 'train_hard_loss' in history and history['train_hard_loss']:
        ax = plt.subplot(num_rows, ncols, plot_index)
        ax.plot(epochs_range, history['train_hard_loss'], 'go-', label='Train Hard Loss')
        ax.plot(epochs_range, history['train_soft_loss'], 'mo-', label='Train Soft Loss (MSE)')
        ax.plot(epochs_range, history['val_loss'], 'ro--', label='Validation Loss') # Val loss es hard loss
        ax.set_title(f'{model_name_tag} - KD Loss Components')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
        plot_index += 1

    # Plot Loss per Pathology (si se calcularon)
    if num_plots_pathology:
         for i in range(num_pathologies):
             # Verificar si hay datos para esta patología
             if i < len(history.get('train_loss_per_pathology', [])) and history['train_loss_per_pathology'][i]:
                  ax = plt.subplot(num_rows, ncols, plot_index)
                  ax.plot(epochs_range, history['train_loss_per_pathology'][i], 'bo-', label=f'Train Loss')
                  ax.plot(epochs_range, history['val_loss_per_pathology'][i], 'ro-', label=f'Val Loss')
                  ax.set_title(f'{model_name_tag} - Loss for {pathology_names[i]}')
                  ax.set_xlabel('Epochs')
                  ax.set_ylabel('Loss')
                  ax.legend()
                  ax.grid(True)
                  plot_index += 1
             else:
                  print(f"WARN: No per-pathology loss data found for {pathology_names[i]} in history.")


    plt.tight_layout(pad=2.0)
    plot_filename = os.path.join(output_dir, f'{model_name_tag}_learning_curves.png')
    try:
        plt.savefig(plot_filename)
        print(f"Learning curves saved to {plot_filename}")
    except Exception as e:
        print(f"ERROR guardando curvas de aprendizaje: {e}")
    plt.close() # Cerrar figura



def plot_auc_roc_curves(all_targets_binary_np, all_probs_present_np, num_pathologies, pathology_names, output_dir, model_name_tag):
    plt.figure(figsize=(10, 8))

    for i in range(num_pathologies):
        # Asegurarse que hay etiquetas positivas y negativas para calcular AUC/ROC
        if len(np.unique(all_targets_binary_np[:, i])) < 2:
            print(f"ADVERTENCIA: No se puede calcular ROC para '{pathology_names[i]}' - solo una clase presente en las etiquetas de test.")
            continue

        fpr, tpr, thresholds = roc_curve(all_targets_binary_np[:, i], all_probs_present_np[:, i])
        auc_score = roc_auc_score(all_targets_binary_np[:, i], all_probs_present_np[:, i])
        plt.plot(fpr, tpr, label=f'{pathology_names[i]} (AUC = {auc_score:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'{model_name_tag} - ROC Curves per Pathology')
    plt.legend(loc="lower right")
    plt.grid(True)
    plot_filename = os.path.join(output_dir, f'{model_name_tag}_auc_roc_curves.png')
    plt.savefig(plot_filename)
    print(f"AUC ROC curves saved to {plot_filename}")
    # plt.show()

# --- 10. Evaluation Function ---
# --- 10. Evaluation Function (Incluye AUC y Reporte adaptado) ---
def evaluate_model(model, dataloader, criterion_bce, criterion_elementwise, device, num_pathologies, pathology_names, output_dir, model_name_tag):
    print(f"\n--- Evaluating {model_name_tag} on Test Set ---")
    # Usar validate_epoch_standard para obtener salidas y targets
    test_loss, test_path_losses, test_outputs, test_targets_20bit = validate_epoch(model_name_tag,
        model, dataloader, criterion_bce, criterion_elementwise, device,
        num_pathologies, pathology_names, 1, 1 # Epoch info no relevante aquí
    )

    print(f"Test Loss (Average BCE): {test_loss:.4f}")
    for i, name in enumerate(pathology_names):
             print(f"  -> Avg Test Loss {name}: {test_path_losses[i]:.4f}")

    # Preparar datos para métricas
    test_outputs_logits_np = test_outputs.cpu().numpy() # Shape: (N, 20)
    test_targets_20bit_np = test_targets_20bit.cpu().numpy() # Shape: (N, 20)

    # --- AUC y Curvas ROC (Presente vs Otros) ---
    try:
        print("\n--- AUC Scores & ROC Curves (Present vs Others) ---")
        plt.figure(figsize=(10, 8))
        auc_scores = []
        valid_auc_classes = 0

        # Calcular probabilidades Sigmoid
        probs_20_np = 1 / (1 + np.exp(-test_outputs_logits_np)) # Sigmoid

        # Obtener targets binarios para PRESENTE (col 0, 4, 8...)
        targets_present_binary_np = test_targets_20bit_np[:, ::NUM_STATES] # (N, 5)
        # Obtener probabilidades para PRESENTE (col 0, 4, 8...)
        probs_present_np = probs_20_np[:, ::NUM_STATES] # (N, 5)

        for i in range(num_pathologies):
            pathology_name = pathology_names[i]
            y_true = targets_present_binary_np[:, i]
            y_score = probs_present_np[:, i]

            if len(np.unique(y_true)) < 2:
                print(f"  {pathology_name}: Skipping ROC/AUC (only one class present)")
                auc_scores.append(np.nan)
                continue

            auc = roc_auc_score(y_true, y_score)
            auc_scores.append(auc)
            valid_auc_classes += 1
            print(f"  {pathology_name}: AUC = {auc:.4f}")

            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            plt.plot(fpr, tpr, lw=2, label=f'{pathology_name} (AUC = {auc:.3f})')

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Chance')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title(f'{model_name_tag} - ROC Curves (Present vs Others)')
        plt.legend(loc="lower right")
        plt.grid(True)
        plot_filename = os.path.join(output_dir, f'{model_name_tag}_auc_roc_curves.png')
        plt.savefig(plot_filename)
        print(f"AUC ROC curves saved to {plot_filename}")
        plt.close()

        if valid_auc_classes > 0:
            print(f"Macro Average AUC: {np.nanmean(auc_scores):.4f}")
        else: print("\nNo valid AUC scores to average.")

    except Exception as e:
        print(f"ERROR calculando AUC/ROC: {e}")

    # --- Classification Report (Estado Predicho 0-3 vs Estado Verdadero 0-3) ---
    try:
        print("\n--- Classification Report (Predicted State 0-3 vs True State 0-3) ---")
        # Obtener índices predichos (argmax sobre bloques de 4 logits)
        predicted_indices_np = np.argmax(test_outputs_logits_np.reshape(-1, num_pathologies, NUM_STATES), axis=2) # (N, 5)
        # Obtener índices verdaderos (argmax sobre bloques de 4 targets)
        true_indices_np = np.argmax(test_targets_20bit_np.reshape(-1, num_pathologies, NUM_STATES), axis=2) # (N, 5)

        label_names_per_state = ["Present", "Absent", "Uncertain", "Not_Mentioned"]
        all_metrics = {}
        print("Calculating metrics per pathology...")
        for i in range(num_pathologies):
            pathology_name = pathology_names[i]
            path_target = true_indices_np[:, i]
            path_pred = predicted_indices_np[:, i]
            print(f"\n--- Metrics for {pathology_name} ---")
            try:
                report = classification_report(
                    path_target,
                    path_pred,
                    target_names=label_names_per_state,
                    labels=np.arange(NUM_STATES),
                    zero_division=0,
                    output_dict=False # Imprimir directo
                )
                print(report)
                acc = accuracy_score(path_target, path_pred)
                print(f"  Overall State Accuracy: {acc:.4f}")
                cm = confusion_matrix(path_target, path_pred, labels=np.arange(NUM_STATES))
                print(f"  Confusion Matrix (Rows: True, Cols: Pred):\n    {label_names_per_state}\n{cm}")
                # Guardar métricas si es necesario
                all_metrics[pathology_name] = classification_report(path_target, path_pred, labels=np.arange(NUM_STATES), output_dict=True, zero_division=0)

            except Exception as e:
                print(f"  Could not generate report/metrics for {pathology_name}: {e}")

    except Exception as e:
        print(f"ERROR calculating classification metrics: {e}")
        all_metrics = None

    print("\n--- Evaluation Finished ---")
    return all_metrics


# --- 11. Main Execution ---
if __name__ == '__main__':
    print("\n--- Initializing Training ---")
    start_overall_time = time.time()

    # # --- Teacher Model Training ---
    # print("\n--- Setting up Teacher Model ---")
    # teacher_model = create_model(args.teacher_model, NUM_OUTPUT_NEURONS, args.use_pretrained)
    # teacher_model.to(DEVICE)
    # # Usar DataParallel si hay múltiples GPUs (opcional)
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs for Teacher Model!")
    #     teacher_model = nn.DataParallel(teacher_model)

    # optimizer_teacher = optim.Adam(teacher_model.parameters(), lr=args.lr)

    # teacher_model, teacher_history = run_training(
    #     f"{args.teacher_model}_Teacher", teacher_model, train_loader, valid_loader, criterion_bce, criterion_bce_elementwise, # Pasar ambas losses BCE
    #     optimizer_teacher, args.epochs, DEVICE, NUM_PATHOLOGIES, args.classes, args.output_dir
    # )
    # plot_learning_curves(teacher_history, NUM_PATHOLOGIES, args.classes, args.output_dir, f"{args.teacher_model}_Teacher")
    # teacher_metrics = evaluate_model(
    #     teacher_model, test_loader, criterion_bce, criterion_bce_elementwise, DEVICE, NUM_PATHOLOGIES, args.classes, args.output_dir, f"{args.teacher_model}_Teacher"
    # )

    # --- Preparación para KD ---
    print("\n\n--- Preparing for Knowledge Distillation ---")
    # Cargar el MEJOR teacher model guardado
    best_teacher_path = os.path.join(args.output_dir, f'{args.teacher_model}_Teacher_best_model.pth')
    if os.path.exists(best_teacher_path):
        print(f"Loading best teacher weights from {best_teacher_path}")
        # Crear una nueva instancia por si DataParallel afecta la carga
        teacher_model_eval = create_model(args.teacher_model, NUM_OUTPUT_NEURONS, use_pretrained=False) # Cargar arquitectura
        # Cargar state_dict manejando DataParallel si fue usado
        state_dict = torch.load(best_teacher_path, map_location=DEVICE)
        try:
             # Quitar prefijo 'module.' si existe (de DataParallel)
             from collections import OrderedDict
             new_state_dict = OrderedDict()
             for k, v in state_dict.items():
                 name = k[7:] if k.startswith('module.') else k
                 new_state_dict[name] = v
             teacher_model_eval.load_state_dict(new_state_dict)
        except: # Si no tenía 'module.' o falla por otra razón
             print("WARN: Could not remove 'module.' prefix, loading state dict directly.")
             teacher_model_eval.load_state_dict(state_dict)

        teacher_model_eval.to(DEVICE)
        teacher_model_eval.eval() # Poner en modo eval para KD
        print("Best teacher model loaded for distillation.")
    else:
        print(f"ERROR: Best teacher model weights not found at {best_teacher_path}. Cannot perform distillation.")
        teacher_model_eval = None # Indicar que no se pudo cargar

    # --- Student Model Training with KD (SOLO si el teacher se cargó) ---
    if teacher_model_eval:
        print("\n\n--- Setting up Student Model for KD ---")
        student_model_kd = create_model(args.student_model, NUM_OUTPUT_NEURONS, args.use_pretrained)

        # best_KD_path = os.path.join(args.output_dir, f'{args.student_model}_Student_KD_best_model.pth')
        # if os.path.exists(best_KD_path):
        #     print(f"Loading best teacher weights from {best_KD_path}")
        #     # Crear una nueva instancia por si DataParallel afecta la carga
        #     teacher_model_eval = create_model(args.teacher_model, NUM_OUTPUT_NEURONS, use_pretrained=False) # Cargar arquitectura
        #     # Cargar state_dict manejando DataParallel si fue usado
        #     state_dict = torch.load(best_KD_path, map_location=DEVICE)
        #     try:
        #         # Quitar prefijo 'module.' si existe (de DataParallel)
        #         from collections import OrderedDict
        #         new_state_dict = OrderedDict()
        #         for k, v in state_dict.items():
        #             name = k[7:] if k.startswith('module.') else k
        #             new_state_dict[name] = v
        #         student_model_kd.load_state_dict(new_state_dict)
        #     except: # Si no tenía 'module.' o falla por otra razón
        #         print("WARN: Could not remove 'module.' prefix, loading state dict directly.")
        #         student_model_kd.load_state_dict(state_dict)

        #     # student_model_kd.to(DEVICE)
        #     # teacher_model_eval.eval() # Poner en modo eval para KD
        #     print("Best KD model loaded for distillation.")
        # else:
        #     print(f"ERROR: Best KD model weights not found at {best_KD_path}. Cannot perform distillation.")
        #     # teacher_model_eval = None # Indicar que no se pudo cargar
            
        student_model_kd.to(DEVICE)
        student_model_kd_parallel = nn.DataParallel(student_model_kd) if torch.cuda.device_count() > 1 else student_model_kd
        if torch.cuda.device_count() > 1: print(f"Using {torch.cuda.device_count()} GPUs for Student KD!")

        optimizer_student_kd = optim.Adam(student_model_kd_parallel.parameters(), lr=args.lr)

        # args.epochs=20
        # Ejecutar entrenamiento con KD
        student_model_kd_trained, student_kd_history = run_training_kd(
            f"{args.student_model}_Student_KD", student_model_kd_parallel, teacher_model_eval, # Pasar teacher cargado
            train_loader, valid_loader,
            criterion_bce_elementwise, criterion_mse, # Pasar losses elementwise y MSE
            optimizer_student_kd, args.epochs, DEVICE, NUM_PATHOLOGIES, args.classes,
            args.output_dir, args.kd_alpha, args.kd_temperature # Pasar alpha y T
        )
        args.epochs=50
        plot_learning_curves(student_kd_history, NUM_PATHOLOGIES, args.classes, args.output_dir, f"{args.student_model}_Student_KD", include_kd_losses=True)
        print("\n--- Evaluating Student Model (trained WITH KD) ---")
        student_kd_metrics = evaluate_model(
            student_model_kd_trained, test_loader, criterion_bce, criterion_bce_elementwise, DEVICE, NUM_PATHOLOGIES, args.classes, args.output_dir, f"{args.student_model}_Student_KD"
        )

    else:
        print("\n--- Skipping Student KD training because Teacher model failed to load. ---")



    # --- Student Model Training ---
    print("\n\n--- Setting up Student Model ---")
    student_model = create_model(args.student_model, NUM_OUTPUT_NEURONS, args.use_pretrained)
    student_model.to(DEVICE)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for Student Model!")
        student_model = nn.DataParallel(student_model)

    optimizer_student = optim.Adam(student_model.parameters(), lr=args.lr)

    student_model, student_history = run_training(
        f"{args.student_model}_Student", student_model, train_loader, valid_loader, criterion_bce, criterion_bce_elementwise,  # Pasar losses elementwise y MS
        optimizer_student, args.epochs, DEVICE, NUM_PATHOLOGIES, args.classes, args.output_dir
    )
    plot_learning_curves(student_history, NUM_PATHOLOGIES, args.classes, args.output_dir, f"{args.student_model}_Student")
    student_metrics = evaluate_model(
        student_model, test_loader, criterion_bce, criterion_bce_elementwise, DEVICE, NUM_PATHOLOGIES, args.classes, args.output_dir, f"{args.student_model}_Student"
    )

    # --- Final Summary ---
    end_overall_time = time.time()
    total_time = end_overall_time - start_overall_time
    print("\n\n--- Experiment Finished ---")
    print(f"Total execution time: {total_time // 60:.0f}m {total_time % 60:.0f}s")
    print(f"Results, models, and plots saved in: {args.output_dir}")
    print("\nComparison:")
    print("- Review the printed evaluation metrics (AUC, Classification Reports) for Teacher, Student_Standard, and Student_KD.")
    print("- Examine the plots saved in the output directory.")
    
    