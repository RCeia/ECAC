# Rodrigo Martins Ceia nº2023222356
# Gabriel Alexandre Sabino Costeira nº 2023222421
import os
import csv
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import random

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ------------------------------
# Constantes de Índices (Features CSV)
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

outdir = ensure_outputs_dir("outputsB")

# ------------------------------
# 0. Funções de Carregamento
# ------------------------------

def carregar_dados_brutos(part_id, pasta_base="FORTH_TRACE_DATASET", devices=(1,2,3,4,5)):
    """
    Carrega os dados originais (raw) para analisar o balanço de amostras.
    Filtra atividades > 7.
    """
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
                        # A atividade é a última coluna. Filtro <= 7
                        if vals[-1] <= 7:
                            rows.append(vals)
                    except ValueError: continue 
        except Exception as e:
            print(f"Erro ao ler ficheiro {path}: {e}")

    return np.array(rows) if rows else None

def load_features_dataset(file_path):
    """
    Carrega o dataset de features (janelas) a partir de um CSV.
    """
    if not os.path.exists(file_path):
        print(f"ERRO: O ficheiro '{file_path}' não existe.")
        print("Execute o script anterior (TP1B) para gerar as features primeiro.")
        return None, None

    print(f"A carregar dataset de features: {file_path} ...")
    
    try:
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
    except Exception as e:
        print(f"Erro ao ler CSV: {e}")
        return None, None

    try:
        data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    except ValueError:
        print("Erro ao carregar dados com numpy.")
        return None, None
        
    return header, data

# ------------------------------
# 1.1 Funções de Visualização (Plots)
# ------------------------------

def plot_class_balance(labels_all, output_path, title="Distribuição", ylabel="Contagem"):
    """
    Gera gráfico de barras com a contagem de amostras por atividade.
    """
    classes_unicas, contagens = np.unique(labels_all, return_counts=True)
    
    print(f"\nContagem por Atividade ({ylabel}):")
    for cls, count in zip(classes_unicas, contagens):
        print(f"Atividade {int(cls)}: {count}")
    
    if len(contagens) > 0:
        max_count = np.max(contagens)
        min_count = np.min(contagens)
        ratio = max_count / min_count if min_count > 0 else float('inf')
        print(f"Ratio Max/Min: {ratio:.2f}")
    
    plt.figure(figsize=(10, 6))
    plt.bar(classes_unicas, contagens, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel("Atividade")
    plt.ylabel(ylabel)
    plt.xticks(classes_unicas)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    savefig(output_path)
    print(f"Gráfico guardado em: {output_path}")


# ------------------------------
# 1.2 SMOTE Implementation
# ------------------------------

def generate_smote_samples(features_data, k_samples, k_neighbors=5):
    """
    Implementa SMOTE (Synthetic Minority Over-sampling Technique).
    Recebe matriz de features e gera k_samples novas.
    """
    n_samples = len(features_data)
    if n_samples < 2:
        print("Erro: Amostras insuficientes para SMOTE.")
        return None

    k_neighbors = min(n_samples - 1, k_neighbors)
    
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(features_data)
    _, indices = nbrs.kneighbors(features_data)
    
    new_samples = []
    
    for _ in range(k_samples):
        idx_base = random.randint(0, n_samples - 1)
        base_sample = features_data[idx_base]
        
        idx_neighbor_local = random.randint(1, k_neighbors)
        neighbor_idx = indices[idx_base][idx_neighbor_local]
        neighbor_sample = features_data[neighbor_idx]
        
        diff = neighbor_sample - base_sample
        gap = random.random()
        synthetic_sample = base_sample + (gap * diff)
        
        new_samples.append(synthetic_sample)
        
    return np.array(new_samples)

# ------------------------------
# 1.3 SMOTE Visualization
# ------------------------------

def plot_smote_visualization(data_participant, synthetic_features, activity_target, feat_names, output_path):
    """
    Gera scatter plot 2D: Dados Reais (Features) vs Sintéticos.
    """
    feat_name_1, feat_name_2 = feat_names
    
    plt.figure(figsize=(10, 7))
    
    unique_labels = np.unique(data_participant[:, IDX_LABEL])
    
    for label in sorted(unique_labels):
        mask_lbl = data_participant[:, IDX_LABEL] == label
        subset = data_participant[mask_lbl]
        feat_subset = subset[:, IDX_FEATS:] 
        
        alpha_val = 0.3 if label != activity_target else 0.6
        label_text = f"Ativ {int(label)}"
        
        # Plot apenas das duas primeiras features
        plt.scatter(feat_subset[:, 0], feat_subset[:, 1], 
                    label=label_text, alpha=alpha_val, s=40)

    plt.scatter(synthetic_features[:, 0], synthetic_features[:, 1],
                color='black', marker='X', s=150, 
                label='Sintéticas (SMOTE)', zorder=10, edgecolors='white')

    plt.title(f"Visualização SMOTE (Ativ {activity_target})\nFeatures: {feat_name_1} vs {feat_name_2}")
    plt.xlabel(feat_name_1)
    plt.ylabel(feat_name_2)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    savefig(output_path)
    print(f"Gráfico SMOTE guardado em: {output_path}")

# ------------------------------
# Main
# ------------------------------

def main():
    # =================================================================
    # EXERCÍCIO 1.1: Análise de Balanço (Dados Brutos / Raw Samples)
    # =================================================================
    print("\n=== 1.1 Análise de Balanço de Classes (Dados Brutos) ===")
    
    PASTA_RAW = "FORTH_TRACE_DATASET"
    NUM_PARTICIPANTES = 15 # Ajuste conforme necessário
    labels_brutos_total = []

    print(f"A ler ficheiros brutos de {PASTA_RAW}...")
    
    for p_id in range(1, NUM_PARTICIPANTES + 1):
        dados_brutos = carregar_dados_brutos(p_id, pasta_base=PASTA_RAW)
        if dados_brutos is not None:
            # A atividade é a última coluna (índice -1 ou 11)
            labels_brutos_total.append(dados_brutos[:, -1])
            
    if len(labels_brutos_total) > 0:
        all_raw_labels = np.concatenate(labels_brutos_total)
        print(f"Total de amostras brutas carregadas: {len(all_raw_labels)}")
        
        path_balance_raw = os.path.join(outdir, "1_1_class_balance_raw_samples.png")
        plot_class_balance(
            all_raw_labels, 
            path_balance_raw, 
            title="Distribuição de Amostras Brutas por Atividade", 
            ylabel="Número de Amostras (Raw)"
        )
    else:
        print("Aviso: Não foram encontrados dados brutos ou pasta incorreta.")


    # =================================================================
    # Carregamento de Features (Para exercícios seguintes)
    # =================================================================
    print("\n=== Carregamento do Dataset de Features ===")
    file_path_feats = os.path.join("outputs", "4_2_features_windows.csv")
    
    header, data_feats = load_features_dataset(file_path_feats)
    if data_feats is None:
        return

    # Filtro de segurança: Apenas atividades <= 7
    mask_le7 = data_feats[:, IDX_LABEL] <= 7
    data_feats = data_feats[mask_le7]
    print(f"Features filtradas (Atividade <= 7): {data_feats.shape[0]} janelas.")

    if data_feats.shape[0] == 0:
        return

    # =================================================================
    # EXERCÍCIO 1.3: SMOTE (Participante 3, Atividade 4)
    # =================================================================
    print("\n=== 1.3 SMOTE: Participante 3, Atividade 4 ===")
    
    PARTICIPANTE_ALVO = 3
    ATIVIDADE_ALVO = 4
    K_SAMPLES = 3
    
    # 1. Filtrar Dados do Participante
    mask_p3 = data_feats[:, IDX_PART] == PARTICIPANTE_ALVO
    data_p3 = data_feats[mask_p3]
    
    if len(data_p3) == 0:
        print(f"Erro: Não há features para o participante {PARTICIPANTE_ALVO}.")
        return

    # 2. Filtrar Atividade Alvo
    mask_act4 = data_p3[:, IDX_LABEL] == ATIVIDADE_ALVO
    data_p3_act4 = data_p3[mask_act4]
    
    if len(data_p3_act4) < 2:
        print(f"Erro: Insuficiente para SMOTE (Part {PARTICIPANTE_ALVO}, Ativ {ATIVIDADE_ALVO}).")
        return

    # 3. Gerar Amostras (SMOTE)
    # Extrair matriz apenas com as features (coluna 2 em diante)
    features_matrix = data_p3_act4[:, IDX_FEATS:]
    
    print(f"Janelas originais disponíveis para SMOTE: {len(features_matrix)}")
    
    synthetic_features = generate_smote_samples(features_matrix, k_samples=K_SAMPLES)
    
    if synthetic_features is not None:
        print(f"Geradas {len(synthetic_features)} novas janelas sintéticas.")
        
        # 4. Visualizar
        feat_name_1 = header[IDX_FEATS]
        feat_name_2 = header[IDX_FEATS + 1]
        path_smote_img = os.path.join(outdir, "1_3_smote_visualization.png")
        
        plot_smote_visualization(
            data_participant=data_p3, 
            synthetic_features=synthetic_features, 
            activity_target=ATIVIDADE_ALVO,
            feat_names=(feat_name_1, feat_name_2),
            output_path=path_smote_img
        )

if __name__ == "__main__":
    main()