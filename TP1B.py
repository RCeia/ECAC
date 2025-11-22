# Rodrigo Martins Ceia nº2023222356
# Gabriel Alexandre Sabino Costeira nº 2023222421
import os
import csv
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import random

# --- Novos Imports para Ex 3 ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report

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
    """
    Versão Robusta: Limpa caracteres '[' e ']' que possam existir no CSV.
    """
    if not os.path.exists(file_path):
        print(f"ERRO: '{file_path}' não existe.")
        return None, None
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        if not lines: return None, None
        
        # 1. Cabeçalho
        header = lines[0].strip().split(',')
        
        # 2. Dados (Parsing manual para remover parêntesis)
        data = []
        for line in lines[1:]:
            line = line.strip()
            if not line: continue
            
            # Remove caracteres problemáticos
            clean_line = line.replace('[', '').replace(']', '').replace('"', '')
            try:
                vals = [float(x) for x in clean_line.split(',')]
                data.append(vals)
            except ValueError:
                continue
                
        return header, np.array(data)

    except Exception as e:
        print(f"Erro ao carregar dataset: {e}")
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
            path_csv = os.path.join(pasta_base, f"part{p_id}", f"part{p_id}dev{dev_id}.csv")
            if not os.path.isfile(path_csv): continue

            try:
                csv_data = np.loadtxt(path_csv, delimiter=',')
            except Exception: continue

            try:
                segments, activities = acc_segmentation(csv_data)
            except Exception as e:
                print(f"Erro na segmentação P{p_id}D{dev_id}: {e}")
                continue

            if not segments: continue

            batch_tensors = []
            batch_meta = [] 

            for seg, act in zip(segments, activities):
                if act > 7: continue 

                seg_30hz, _ = resample_to_30hz_5s(seg, fs_original)
                batch_tensors.append(seg_30hz)
                batch_meta.append([p_id, act])
            
            if not batch_tensors: continue

            x_np = np.array(batch_tensors) # (B, 150, 3)
            x_np = np.transpose(x_np, (0, 2, 1)) # (B, 3, 150)
            
            MINI_BATCH = 32
            n_total = len(x_np)
            
            with torch.no_grad():
                for i in range(0, n_total, MINI_BATCH):
                    x_batch = x_np[i : i + MINI_BATCH]
                    x_tensor = torch.from_numpy(x_batch).float().to(device)
                    emb_batch = encoder(x_tensor).cpu().numpy() 

                    for j, emb_vec in enumerate(emb_batch):
                        meta = batch_meta[i + j]
                        all_rows.append( meta + emb_vec.tolist() )
            
            print(f" -> P{p_id} Dev{dev_id}: {len(batch_tensors)} segmentos processados.")

    if all_rows:
        n_dim = len(all_rows[0]) - 2
        header = ["participante", "label"] + [f"emb_{k}" for k in range(n_dim)]
        save_csv(output_csv, header, all_rows)
        print(f"\nSUCESSO: {len(all_rows)} embeddings guardados em '{output_csv}'.")
    else:
        print("Aviso: Nenhum embedding gerado.")

# =============================================================================
# EXERCÍCIO 3: SPLITTING E PIPELINE
# =============================================================================

# --- Função ReliefF (necessária para o Cenário C) ---
def reliefF(X, y, n_neighbors=10, n_samples=200):
    """
    Implementação simplificada do ReliefF para seleção de features.
    """
    rng = np.random.RandomState(0)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    n, d = X.shape

    # Amostrar um subconjunto para eficiência
    m = min(n_samples, n)
    idx_s = rng.choice(n, m, replace=False)

    scores = np.zeros(d)
    
    # Nearest Neighbors global para encontrar hits/misses
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(X)
    
    for i in idx_s:
        xi, yi = X[i], y[i]
        
        # Encontrar vizinhos no dataset todo
        dists, inds = nbrs.kneighbors([xi])
        # inds[0][0] é o próprio ponto, então pegamos do 1 em diante
        neighbors_idx = inds[0][1:]
        
        # Calcular update de scores
        # (Simplificação: ReliefF clássico busca k hits e k misses separadamente.
        #  Aqui usamos vizinhança geral e penalizamos/premiamos conforme a classe)
        
        for n_idx in neighbors_idx:
            xn = X[n_idx]
            yn = y[n_idx]
            
            dist_feat = np.abs(xi - xn)
            
            # Normalizar diff se possível (assumindo dados normalizados)
            
            if yi == yn: # Hit (mesma classe) -> penaliza features que são diferentes
                scores -= dist_feat
            else: # Miss (classe diferente) -> premia features que são diferentes
                scores += dist_feat

    return scores


# --- 3.1 Within-Subject Split ---
def split_within_subjects(X, y, participants, seed=42):
    """
    Divide 60% Treino, 20% Val, 20% Teste, misturando dados de todos os participantes.
    """
    # 1. Separa Treino (60%) do Resto (40%)
    X_train, X_temp, y_train, y_temp, p_train, p_temp = train_test_split(
        X, y, participants, test_size=0.4, random_state=seed, stratify=y
    )
    # 2. Separa Resto em Validação (metade de 40% -> 20%) e Teste (20%)
    X_val, X_test, y_val, y_test, p_val, p_test = train_test_split(
        X_temp, y_temp, p_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )
    
    print(f"   [Split 3.1 Within] Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test

# --- 3.2 Between-Subject Split ---
def split_between_subjects(X, y, participants, seed=42):
    """
    Divide por participante: 9 Treino, 3 Validação, 3 Teste.
    """
    unique_parts = np.unique(participants)
    np.random.seed(seed)
    np.random.shuffle(unique_parts)
    
    n_total = len(unique_parts)
    if n_total < 3: return None
        
    n_train = 9 if n_total >= 15 else int(n_total * 0.6)
    n_val = 3 if n_total >= 15 else int(n_total * 0.2)
    
    train_ids = unique_parts[:n_train]
    val_ids = unique_parts[n_train : n_train + n_val]
    test_ids = unique_parts[n_train + n_val :]
    
    mask_train = np.isin(participants, train_ids)
    mask_val = np.isin(participants, val_ids)
    mask_test = np.isin(participants, test_ids)
    
    print(f"   [Split 3.2 Between] Train IDs: {train_ids} | Val: {val_ids} | Test: {test_ids}")
    return X[mask_train], y[mask_train], X[mask_val], y[mask_val], X[mask_test], y[mask_test]


# --- 3.4 Pipeline ---
def process_pipeline_scenarios(X_train, X_val, X_test, y_train):
    """
    Gera os cenários A (Normal), B (PCA 90%) e C (ReliefF Top 15).
    """
    results = {}
    
    # 0. Normalização (Fit apenas no Treino)
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)
    X_test_norm = scaler.transform(X_test)
    
    # Cenário A: All Features (Normalizado)
    results['A'] = (X_train_norm, X_val_norm, X_test_norm)
    
    # Cenário B: PCA (90% Variância)
    # Nota: fit apenas no treino normalizado
    pca = PCA(n_components=0.90, random_state=42)
    X_train_pca = pca.fit_transform(X_train_norm)
    X_val_pca = pca.transform(X_val_norm)
    X_test_pca = pca.transform(X_test_norm)
    
    results['B'] = (X_train_pca, X_val_pca, X_test_pca)
    print(f"      -> Cenário B (PCA): {X_train.shape[1]} dims -> {pca.n_components_} componentes.")
    
    # Cenário C: ReliefF (Top 15 Features)
    # Usar apenas o treino para calcular scores
    print("      -> Cenário C (ReliefF): A calcular scores (pode demorar)...")
    scores = reliefF(X_train_norm, y_train, n_neighbors=10, n_samples=300)
    
    # Selecionar Top 15 índices
    top_15_idx = np.argsort(scores)[::-1][:15]
    
    X_train_relief = X_train_norm[:, top_15_idx]
    X_val_relief = X_val_norm[:, top_15_idx]
    X_test_relief = X_test_norm[:, top_15_idx]
    
    results['C'] = (X_train_relief, X_val_relief, X_test_relief)
    print(f"      -> Cenário C (ReliefF): Selecionadas as 15 melhores features.")
    
    return results

# =============================================================================
# EXERCÍCIO 4: MODEL LEARNING (k-NN)
# =============================================================================

# --- 4.1 Implementação Customizada do k-NN ---
class MyKNN:
    """
    Implementação manual de k-NN para satisfazer o requisito 4.1.
    Usa distância euclidiana e votação por maioria.
    """
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """Armazena os dados de treino."""
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X_test):
        """Calcula distâncias e devolve predições."""
        predictions = []
        X_test = np.array(X_test)
        
        for row_test in X_test:
            # 1. Distância Euclidiana (Broadcasting)
            # sqrt(sum((x_train - x_test)^2))
            dists = np.linalg.norm(self.X_train - row_test, axis=1)
            
            # 2. Encontrar os índices dos k vizinhos mais próximos (menores distâncias)
            k_indices = np.argsort(dists)[:self.k]
            
            # 3. Obter as labels desses vizinhos
            k_nearest_labels = self.y_train[k_indices]
            
            # 4. Votação por maioria
            unique, counts = np.unique(k_nearest_labels, return_counts=True)
            most_common = unique[np.argmax(counts)]
            predictions.append(most_common)
            
        return np.array(predictions)

# --- 4.2 Métricas de Avaliação ---
def calculate_metrics(y_true, y_pred, set_name="Validation"):
    """
    Calcula e imprime métricas de classificação: Accuracy, F1, Precision, Recall e Matriz Confusão.
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n   --- Métricas [{set_name}] ---")
    print(f"   Accuracy:  {acc:.4f}")
    print(f"   F1-Score:  {f1:.4f} (Weighted)")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall:    {rec:.4f}")
    print(f"   Matriz de Confusão:\n{cm}")
    
    return acc, f1

# ------------------------------
# Main
# ------------------------------

def main():
    # --- EXERCÍCIO 1.1 ---
    print("\n=== 1.1 Balanço de Classes (Dados Brutos) ===")
    PASTA_RAW = "FORTH_TRACE_DATASET"
    PARTICIPANTES = list(range(0, 15)) # 0 a 14
    
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
        print(f"Ficheiro '{path_embed}' já existe. A saltar extração.")

    # =================================================================
    # EXERCÍCIO 3: PIPELINE (FEATURES + EMBEDDINGS)
    # =================================================================
    print("\n=== 3. Pipeline (Splitting & PCA) ===")
    

    # Preparar Datasets
    # data (Features) já está carregado. Carregamos Embeddings.
    data_features = data 
    _, data_embeds = load_features_dataset(path_embed)
    
    if data_embeds is not None:
        data_embeds = data_embeds[data_embeds[:, IDX_LABEL] <= 7]

    datasets_to_process = []
    if data_features is not None: datasets_to_process.append(("FEATURES", data_features))
    if data_embeds is not None: datasets_to_process.append(("EMBEDDINGS", data_embeds))

    ready_data = {} # Guarda dados prontos para treino

    for name, ds in datasets_to_process:
        print(f"\n>>> Processando {name} ({len(ds)} amostras)...")
        
        X = ds[:, IDX_FEATS:]
        y = ds[:, IDX_LABEL]
        p = ds[:, IDX_PART]
        
        # 3.1 Split Within-Subjects
        print(" -> Estratégia 3.1 (Within):")
        split_within_subjects(X, y, p)
        
        # 3.2 Split Between-Subjects (Preferida)
        print(" -> Estratégia 3.2 (Between):")
        split_res = split_between_subjects(X, y, p)
        
        if split_res is not None:
            X_tr, y_tr, X_val, y_val, X_te, y_te = split_res
            
            # 3.4 Pipeline (Aplicado apenas à estratégia 3.2)
            print(" -> Aplicando Pipeline à Estratégia 3.2:")
            scenarios = process_pipeline_scenarios(X_tr, X_val, X_te, y_tr)
            
            ready_data[name] = {
                'y': (y_tr, y_val, y_te),
                'scenarios': scenarios
            }
        
    print("\nPipeline concluída! Dados prontos em 'ready_data'.")

    # =================================================================
    # EXERCÍCIO 4: MODEL LEARNING (k-NN)
    # =================================================================
    print("\n=== 4. Model Learning (k-NN) ===")
    
    # Parâmetro k do vizinho
    K_NEIGHBORS = 5
    
    for ds_name in ready_data:
        print(f"\n>>> Avaliando Modelos para: {ds_name}")
        
        # Recuperar dados comuns (labels)
        y_train, y_val, y_test = ready_data[ds_name]['y']
        
        # Iterar sobre cenários (A: All Features, B: PCA)
        scenarios = ready_data[ds_name]['scenarios']
        
        for sc_name, sc_data in scenarios.items():
            if sc_data is None: continue # Pula se for None (ex: C não implementado)
            
            print(f"\n   --- Cenário {sc_name} ---")
            X_train, X_val, X_test = sc_data
            
            # Treino
            # Nota: Usamos sklearn para rapidez, mas a classe MyKNN está implementada acima.
            # Se quiser usar a customizada: clf = MyKNN(k=K_NEIGHBORS)
            clf = KNeighborsClassifier(n_neighbors=K_NEIGHBORS)
            clf.fit(X_train, y_train)
            
            # Predição (Validação)
            y_pred_val = clf.predict(X_val)
            
            # 4.2 Métricas
            calculate_metrics(y_val, y_pred_val, set_name=f"Validation - {sc_name}")

    print("\nPipeline Final Concluída!")
if __name__ == "__main__":
    main()