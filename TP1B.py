# Rodrigo Martins Ceia nº2023222356
# Gabriel Alexandre Sabino Costeira nº 2023222421
import os
import csv
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import random

# --- Imports para Embeddings (Exercício 2) ---
try:
    import torch
    # Importar funções do ficheiro auxiliar
    from embeddings_extractor import load_model, resample_to_30hz_5s, acc_segmentation
    HAS_TORCH = True
except ImportError as e:
    HAS_TORCH = False
    print(f"AVISO: Falha ao importar torch ou embeddings_extractor: {e}")
    print("O Exercício 2 será ignorado.")

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ------------------------------
# Constantes
# ------------------------------
IDX_PART = 0
IDX_LABEL = 1
IDX_FEATS = 2

# ------------------------------
# Utilitários de saída
# ------------------------------

def ensure_outputs_dir(dirname="outputs"):
    os.makedirs(dirname, exist_ok=True)
    return dirname

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def save_csv(path, header, rows):
    ensure_outputs_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if header: writer.writerow(header)
        writer.writerows(rows)

outdir = ensure_outputs_dir("outputsB")

# ------------------------------
# 0. Funções de Carregamento
# ------------------------------

def carregar_dados_brutos(part_id, pasta_base="FORTH_TRACE_DATASET", devices=(1,2,3,4,5)):
    rows = []
    for dev in devices:
        path = os.path.join(pasta_base, f"part{part_id}", f"part{part_id}dev{dev}.csv")
        if not os.path.isfile(path): continue
        try:
            with open(path, "r") as f:
                reader = csv.reader(f)
                for line in reader:
                    if not line: continue
                    try:
                        vals = [float(x) for x in line]
                        if vals[-1] <= 7: rows.append(vals)
                    except ValueError: continue 
        except Exception as e:
            print(f"Erro ao ler {path}: {e}")
    return np.array(rows) if rows else None

def load_features_dataset(file_path):
    if not os.path.exists(file_path):
        print(f"ERRO: '{file_path}' não existe.")
        return None, None
    try:
        with open(file_path, 'r') as f:
            header = next(csv.reader(f))
        data = np.loadtxt(file_path, delimiter=',', skiprows=1)
        return header, data
    except Exception as e:
        print(f"Erro ao carregar features: {e}")
        return None, None

# ------------------------------
# 1.1 Plots
# ------------------------------
def plot_class_balance(labels_all, output_path, title="Distribuição", ylabel="Contagem"):
    classes, counts = np.unique(labels_all, return_counts=True)
    print(f"\nContagem ({ylabel}):")
    for c, n in zip(classes, counts): print(f"Ativ {int(c)}: {n}")
    
    if len(counts) > 0:
        r = np.max(counts) / np.min(counts)
        print(f"Ratio Max/Min: {r:.2f}")

    plt.figure(figsize=(10, 6))
    plt.bar(classes, counts, color='skyblue', edgecolor='black')
    plt.title(title); plt.xlabel("Atividade"); plt.ylabel(ylabel)
    plt.xticks(classes); plt.grid(axis='y', linestyle='--', alpha=0.7)
    savefig(output_path)

# ------------------------------
# 1.2 e 1.3 SMOTE Logic
# ------------------------------
def generate_smote_samples(features_data, k_samples, k_neighbors=5):
    if len(features_data) < 2: return None
    k_neighbors = min(len(features_data) - 1, k_neighbors)
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(features_data)
    _, indices = nbrs.kneighbors(features_data)
    
    new_samples = []
    for _ in range(k_samples):
        base = features_data[random.randint(0, len(features_data) - 1)]
        neighbor = features_data[indices[random.randint(0, len(features_data) - 1)][random.randint(1, k_neighbors)]]
        new_samples.append(base + random.random() * (neighbor - base))
    return np.array(new_samples)

def plot_smote_visualization(data_participant, synthetic_features, activity_target, feat_names, output_path):
    f1, f2 = feat_names
    plt.figure(figsize=(10, 7))
    for lbl in np.unique(data_participant[:, IDX_LABEL]):
        subset = data_participant[data_participant[:, IDX_LABEL] == lbl][:, IDX_FEATS:]
        alpha = 0.3 if lbl != activity_target else 0.6
        plt.scatter(subset[:, 0], subset[:, 1], label=f"Ativ {int(lbl)}", alpha=alpha, s=40)
    
    plt.scatter(synthetic_features[:, 0], synthetic_features[:, 1], c='black', marker='X', s=150, label='SMOTE', edgecolors='white')
    plt.title(f"SMOTE (Ativ {activity_target}) - {f1} vs {f2}")
    plt.xlabel(f1); plt.ylabel(f2); plt.legend(); plt.grid(True, alpha=0.5)
    savefig(output_path)

# =============================================================================
# EXERCÍCIO 2.1: EMBEDDINGS EXTRACTOR (Opção C - Stacking)
# =============================================================================

def gerar_embeddings_dataset(participantes, pasta_base, output_csv):
    """
    Extrai embeddings usando 'embeddings_extractor.py' (harnet5).
    Usa Opção C: Stacking (cada device é tratado como uma sample independente).
    """
    if not HAS_TORCH: return

    print("A carregar modelo de embeddings (harnet5)...")
    try:
        encoder = load_model() # Importado
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder.to(device)
    
    print(f"A extrair embeddings (Stacking) em {device}...")
    fs_original = 50.0
    all_rows = []

    # Loop: Participantes -> Devices
    for p_id in participantes:
        for dev_id in [1, 2, 3, 4, 5]:
            # 1. Carregar CSV completo deste device (usando numpy loadtxt direto é mais rápido aqui)
            path_csv = os.path.join(pasta_base, f"part{p_id}", f"part{p_id}dev{dev_id}.csv")
            if not os.path.isfile(path_csv): continue

            try:
                # Carregar raw data para passar ao segmentador
                csv_data = np.loadtxt(path_csv, delimiter=',')
            except Exception: continue

            # 2. Segmentação (usa função importada)
            # Retorna lista de segmentos (N, 3) e lista de atividades
            try:
                segments, activities = acc_segmentation(csv_data)
            except Exception as e:
                print(f"Erro na segmentação P{p_id}D{dev_id}: {e}")
                continue

            if not segments: continue

            # 3. Reamostragem e Preparação de Batch
            # Filtra atividades > 7 e prepara tensores
            batch_tensors = []
            batch_meta = [] # (part_id, label)

            for seg, act in zip(segments, activities):
                if act > 7: continue # Filtro <= 7

                # Resample para 30Hz (retorna (dados, fs))
                seg_30hz, _ = resample_to_30hz_5s(seg, fs_original)
                batch_tensors.append(seg_30hz)
                batch_meta.append([p_id, act])
            
            if not batch_tensors: continue

            # 4. Inferência em Batch
            # Formato harnet5: (Batch, Channels=3, Time=150)
            x_np = np.array(batch_tensors) # (B, 150, 3)
            x_np = np.transpose(x_np, (0, 2, 1)) # (B, 3, 150)
            
            # Processar em mini-batches para não estourar memória
            MINI_BATCH = 32
            n_total = len(x_np)
            
            with torch.no_grad():
                for i in range(0, n_total, MINI_BATCH):
                    x_batch = x_np[i : i + MINI_BATCH]
                    x_tensor = torch.from_numpy(x_batch).float().to(device)
                    
                    emb_batch = encoder(x_tensor).cpu().numpy() # (B, D_embed)

                    # Guardar linhas
                    for j, emb_vec in enumerate(emb_batch):
                        meta = batch_meta[i + j]
                        # Linha: [Part, Label, Emb_0, Emb_1...]
                        all_rows.append( meta + emb_vec.tolist() )
            
            print(f" -> P{p_id} Dev{dev_id}: {len(batch_tensors)} segmentos processados.")

    # 5. Guardar CSV
    if all_rows:
        n_dim = len(all_rows[0]) - 2
        header = ["participante", "label"] + [f"emb_{k}" for k in range(n_dim)]
        save_csv(output_csv, header, all_rows)
        print(f"\nSUCESSO: {len(all_rows)} embeddings guardados em '{output_csv}'.")
    else:
        print("Aviso: Nenhum embedding gerado.")

# ------------------------------
# Main
# ------------------------------

def main():
    # --- EXERCÍCIO 1.1 ---
    print("\n=== 1.1 Balanço de Atividades (Dados Brutos) ===")
    PASTA_RAW = "FORTH_TRACE_DATASET"
    PARTICIPANTES = list(range(0, 15)) # 1 a 15
    
    raw_labels = []
    for p in PARTICIPANTES:
        d = carregar_dados_brutos(p, PASTA_RAW)
        if d is not None: raw_labels.append(d[:, -1])
    
    if raw_labels:
        all_labels = np.concatenate(raw_labels)
        plot_class_balance(all_labels, os.path.join(outdir, "1_1_balance_raw.png"), "Distribuição Raw")
    
    # --- EXERCÍCIO 1.3 (SMOTE) ---
    print("\n=== 1.3 SMOTE (Features) ===")
    path_feats = os.path.join("outputs", "4_2_features_windows.csv")
    h, data = load_features_dataset(path_feats)
    
    if data is not None:
        data = data[data[:, IDX_LABEL] <= 7] # Filtro
        
        # P3, Ativ 4
        p3_data = data[data[:, IDX_PART] == 3]
        p3_act4 = p3_data[p3_data[:, IDX_LABEL] == 4]
        
        if len(p3_act4) > 1:
            feats = p3_act4[:, IDX_FEATS:]
            synth = generate_smote_samples(feats, k_samples=3)
            if synth is not None:
                plot_smote_visualization(p3_data, synth, 4, (h[IDX_FEATS], h[IDX_FEATS+1]), os.path.join(outdir, "1_3_smote.png"))
        else:
            print("Dados insuficientes para SMOTE (P3, Ativ4).")

    # --- EXERCÍCIO 2.1 (Embeddings) ---
    print("\n=== 2.1 Embeddings Dataset (Stacking) ===")
    path_embed = os.path.join(outdir, "EMBEDDINGS_DATASET.csv")
    
    if not os.path.exists(path_embed):
        gerar_embeddings_dataset(PARTICIPANTES, PASTA_RAW, path_embed)
    else:
        print(f"Ficheiro '{path_embed}' já existe. A saltar.")

if __name__ == "__main__":
    main()