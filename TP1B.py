# Rodrigo Martins Ceia nº2023222356
# Gabriel Alexandre Sabino Costeira nº 2023222421
import os
import csv
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import random
from scipy.stats import ttest_rel, skew, kurtosis, entropy
from scipy import stats
# Adicionar junto aos outros imports do sklearn
from sklearn.ensemble import RandomForestClassifier

# Adicionar aos imports existentes do sklearn
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

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

# -------------
# Exercício 1.1
# -------------

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

# --------------------
# Exercícios 1.2 e 1.3
# --------------------

# SMOTE Logic

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

# -------------
# Exercício 2.1
# -------------

# EMBEDDINGS EXTRACTOR (Opção C - Stacking)

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

# -----------
# Exercício 3
# -----------

# SPLITTING E PIPELINE

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

# -------------
# Exercício 3.1
# -------------

# Within-Subject Split

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

# -------------
# Exercício 3.2
# -------------

# Between-Subject Split 

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

# -------------
# Exercício 3.4
# -------------

# Pipeline

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

# -----------
# Exercício 4
# -----------

# MODEL LEARNING (k-NN)

#--------------
# Exercício 4.1 
#--------------

# Implementação Customizada do k-NN

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

# -------------
# EXERCÍCIO 5.1
# -------------

#  MODEL LEARNING (k-NN)

def tune_and_retrain(X_train, y_train, X_val, y_val, X_test, y_test, k_values=[1,3,5,7,9,11,13]):
    """
    Exercício 5.1:
    1. Tuning: Encontrar melhor k usando Validação.
    2. Retrain: Treinar com (Treino + Validação) usando melhor k.
    3. Avaliação: Testar no conjunto de Teste.
    """
    # A. Tuning
    best_acc = -1
    best_k = k_values[0]
    
    for k in k_values:
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train)
        # Score devolve a accuracy diretamente
        acc = clf.score(X_val, y_val)
        
        if acc > best_acc:
            best_acc = acc
            best_k = k
            
    # B. Retrain (Juntar dados)
    X_full_train = np.vstack((X_train, X_val))
    y_full_train = np.concatenate((y_train, y_val))
    
    # C. Modelo Final
    final_clf = KNeighborsClassifier(n_neighbors=best_k)
    final_clf.fit(X_full_train, y_full_train)
    
    # D. Avaliação Final
    y_pred_test = final_clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    # DEVOLVE 4 VALORES: Accuracy, K, True Labels, Predicted Labels
    return test_acc, best_k, y_test, y_pred_test


def perform_hypothesis_testing(results_dict):
    """
    5.3: Comparação Estatística usando Paired T-Test.
    Recebe: {'ModelName': [acc_iter1, acc_iter2, ...], ...}
    """
    print("\n" + "="*60)
    print("=== 5.3 TESTES DE HIPÓTESE (Statistical Significance) ===")
    print("="*60)
    
    if len(results_dict) < 2:
        print("Menos de 2 modelos para comparar. Ignorando.")
        return

    # 1. Determinar o Melhor Modelo (Baseline)
    model_means = {m: np.mean(s) for m, s in results_dict.items()}
    best_model_name = max(model_means, key=model_means.get)
    best_scores = results_dict[best_model_name]
    
    print(f"🏆 MELHOR MODELO (Baseline): {best_model_name}")
    print(f"   Média Accuracy: {model_means[best_model_name]:.4f}")
    print(f"   Desvio Padrão:  {np.std(best_scores):.4f}")
    print("-" * 60)
    print(f"{'Modelo Comparado':<30} | {'Média':<8} | {'p-value':<10} | {'Significativo?'}")
    print("-" * 60)

    # 2. Comparar o Melhor contra os Restantes
    alpha = 0.05 # Nível de significância (95% confiança)
    
    for model_name, scores in results_dict.items():
        if model_name == best_model_name:
            continue
            
        # Teste T Emparelhado (Paired T-Test)
        # Usamos emparelhado porque ambos os modelos foram testados 
        # EXATAMENTE nas mesmas divisões (seeds) de dados.
        t_stat, p_val = ttest_rel(best_scores, scores)
        
        is_significant = "SIM ✅" if p_val < alpha else "NÃO ❌"
        
        print(f"{model_name:<30} | {np.mean(scores):.4f}   | {p_val:.2e}   | {is_significant}")

    print("-" * 60)
    print("Justificação Estatística:")
    print("Utilizou-se o 'Paired Samples t-test' (Teste T Emparelhado).")
    print("Justificação: As amostras de performance não são independentes, pois todos os")
    print("modelos foram avaliados sobre as mesmas repetições de splits (mesmas seeds).")
    print("Isto reduz a variância explicada pelo split, focando na diferença real dos modelos.")
    print("="*60 + "\n")


def plot_results_distribution(results_dict, output_dir):
    """
    Gera um Boxplot com os pontos individuais (jitter) sobrepostos
    para visualizar a distribuição estatística da Accuracy.
    """
    print("\n   -> A gerar gráfico de distribuição de resultados...")
    
    models = list(results_dict.keys())
    data = [results_dict[m] for m in models]
    
    plt.figure(figsize=(12, 7))
    
    # 1. Boxplot (Mostra a mediana e quartis)
    plt.boxplot(data, labels=models, showfliers=False, patch_artist=True, 
                boxprops=dict(facecolor='lightblue', alpha=0.5))
    
    # 2. Scatter Plot (Mostra os 20 pontos reais de cada modelo)
    # Adicionamos um pequeno ruído aleatório no eixo X (jitter) para os pontos não ficarem sobrepostos
    for i, model_scores in enumerate(data):
        x = np.random.normal(i + 1, 0.04, size=len(model_scores))
        plt.plot(x, model_scores, 'r.', alpha=0.6)
        
    plt.title(f"Distribuição de Performance (Accuracy) - {len(data[0])} Repetições")
    plt.ylabel("Accuracy")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    
    filename = os.path.join(output_dir, "5_distribuicao_resultados.png")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"      Gráfico guardado em: {filename}")


# -----------
# Exercício 6
# -----------

# DEPLOYMENT (Classificação de Novos Dados Brutos)

# --- Funções Auxiliares de Extração (Versão compacta para Produção) ---
def _deploy_time_features(x):
    if len(x) == 0: return [0.0]*10
    diff_sign = np.sign(x[1:]) != np.sign(x[:-1])
    zcr = float(np.sum(diff_sign)) / (len(x) - 1 + 1e-12)
    # Ordem: mean, std, var, rms, median, min, max, skew, kurt, zcr
    return [float(np.mean(x)), float(np.std(x)), float(np.var(x)), 
            float(np.sqrt(np.mean(x**2))), float(np.median(x)), float(np.min(x)), 
            float(np.max(x)), float(skew(x)), float(kurtosis(x)), zcr]

def _deploy_spec_features(x, fs=50.0):
    # Se o sinal for vazio, retorna 10 zeros (para manter consistência das 60 features totais)
    if len(x) == 0: return [0.0]*10 
    
    # Preparação básica (Janela Hanning + FFT)
    w = np.hanning(len(x))
    X_fft = np.fft.rfft((x - np.mean(x)) * w)
    mag = np.abs(X_fft)
    psd = mag**2
    psd_sum = np.sum(psd) + 1e-12
    freqs = np.fft.rfftfreq(len(x), d=1.0/fs)
    
    # 1. Centroid
    centroid = np.sum(freqs * psd) / psd_sum
    
    # 2. Bandwidth
    var = np.sum(((freqs - centroid)**2) * psd) / psd_sum
    bandwidth = np.sqrt(max(var, 0))
    
    # 3. Peak Freq
    peak_freq = float(freqs[np.argmax(psd)])
    
    # 4. Entropy
    spec_ent = float(entropy(psd/psd_sum))
    
    # 5. Flatness
    gmean = np.exp(np.mean(np.log(psd + 1e-12)))
    amean = np.mean(psd) + 1e-12
    flatness = float(gmean / amean)
    
    # --- FEATURES QUE FALTAVAM ---
    
    # 6. Rolloff 85
    cumulative_psd = np.cumsum(psd)
    rolloff_threshold = 0.85 * psd_sum
    idx_rolloff = np.where(cumulative_psd >= rolloff_threshold)[0]
    rolloff_85 = float(freqs[idx_rolloff[0]]) if len(idx_rolloff) > 0 else 0.0
    
    # 7. Crest Factor
    crest = float(np.max(mag) / (np.mean(mag) + 1e-12))
    
    # 8. Contrast (Simplificado em 6 bandas)
    n_bands = 6
    if len(psd) >= n_bands:
        band_edges = np.linspace(0, len(psd), n_bands + 1, dtype=int)
        band_max = [np.max(psd[band_edges[i]:band_edges[i+1]]) for i in range(n_bands)]
        band_min = [np.min(psd[band_edges[i]:band_edges[i+1]]) for i in range(n_bands)]
        contrast = float(np.mean(band_max) - np.mean(band_min))
    else:
        contrast = 0.0
        
    # 9. Energy
    energy = float(psd_sum)
    
    # 10. Spread (Na sua implementação original do TP1A, era idêntico ao bandwidth)
    spread = bandwidth 

    # Retorna lista com 10 elementos
    return [centroid, bandwidth, peak_freq, spec_ent, flatness, 
            rolloff_85, crest, contrast, energy, spread]

def _deploy_extract_feats(raw_window):
    """
    Recebe janela bruta (N, 9) -> [Ax, Ay, Az, Gx, Gy, Gz, Mx, My, Mz]
    Calcula Módulo -> Extrai Features Estatísticas.
    """
    feats_vec = []
    # 1. Calcular Módulos (Raiz Quadrada da Soma dos Quadrados)
    acc_mod = np.sqrt(np.sum(raw_window[:, 0:3]**2, axis=1))
    gyr_mod = np.sqrt(np.sum(raw_window[:, 3:6]**2, axis=1))
    mag_mod = np.sqrt(np.sum(raw_window[:, 6:9]**2, axis=1))
    
    # 2. Extrair features para cada sensor
    for signal in [acc_mod, gyr_mod, mag_mod]:
        feats_vec.extend(_deploy_time_features(signal))
        feats_vec.extend(_deploy_spec_features(signal))
        
    return np.array(feats_vec).reshape(1, -1) # Retorna shape (1, n_features)

class ActivityDeployer:
    """
    O 'Modelo Final' pronto a usar. Guarda o Scaler, o Classificador e o PCA.
    """
    def __init__(self, scaler, classifier, pca=None):
        self.scaler = scaler
        self.classifier = classifier
        self.pca = pca
        
    def predict(self, raw_data_matrix):
        """
        Processa dados brutos e devolve a atividade prevista.
        Input: Matriz numpy (N, 9)
        """
        # 1. Ajustar tamanho da janela para 250 (5 segundos @ 50Hz)
        # Se tiver 256, cortamos as últimas 6.
        target_len = 250
        if len(raw_data_matrix) >= target_len:
            window = raw_data_matrix[:target_len, :]
        else:
            # Padding com zeros se o ficheiro for muito curto (segurança)
            pad = np.zeros((target_len - len(raw_data_matrix), 9))
            window = np.vstack([raw_data_matrix, pad])
            
        # 2. Extração de Features
        X_features = _deploy_extract_feats(window)
        
        # 3. Normalização
        X_norm = self.scaler.transform(X_features)
        
        # 4. PCA (se aplicável)
        if self.pca is not None:
            X_final = self.pca.transform(X_norm)
        else:
            X_final = X_norm
            
        # 5. Classificação
        prediction = self.classifier.predict(X_final)
        return int(prediction[0])

def train_deployable_model(X, y, best_k, use_pca=False):
    """
    Treina o modelo com TODOS os dados disponíveis (sem splits) para máxima performance.
    """
    print(f"\n[Deploy] A treinar modelo final (k={best_k}, PCA={use_pca})...")
    
    # 1. Treinar Scaler com tudo
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    
    X_train_final = X_norm
    pca_model = None
    
    # 2. Treinar PCA com tudo (se pedido)
    if use_pca:
        pca_model = PCA(n_components=0.90, random_state=42)
        X_train_final = pca_model.fit_transform(X_norm)
        print(f"[Deploy] PCA treinado. Componentes mantidos: {pca_model.n_components_}")
        
    # 3. Treinar k-NN com tudo
    clf = KNeighborsClassifier(n_neighbors=best_k)
    clf.fit(X_train_final, y)
    
    return ActivityDeployer(scaler, clf, pca_model)


# -----------
# Exercício 7 
# -----------

# OTIMIZAÇÃO (Outliers & Data Augmentation) - VERSÃO ROBUSTA

def remove_outliers_zscore_per_class(X, y, k=3):
    """
    Remove outliers utilizando Z-Score (k=3) individualmente por classe.
    Versão robusta contra NaNs e colunas com desvio padrão zero.
    """
    X_clean_list = []
    y_clean_list = []
    
    classes = np.unique(y)
    
    for c in classes:
        mask_c = (y == c)
        X_c = X[mask_c]
        
        # Se houver poucos dados, não removemos nada (segurança)
        if len(X_c) < 5:
            X_clean_list.append(X_c)
            y_clean_list.append(y[mask_c])
            continue

        # Calcular Z-Score
        # Se uma coluna tiver desvio padrão 0, o zscore dá NaN.
        # nan_policy='omit' pode não ser suficiente, tratamos manualmente.
        with np.errstate(divide='ignore', invalid='ignore'):
            z_scores = stats.zscore(X_c, axis=0)
        
        # Substituir NaNs (que vêm de divisão por zero) por 0.0
        # Isto significa: "se a feature é constante, não é outlier"
        z_scores = np.nan_to_num(z_scores, nan=0.0)
        z_scores = np.abs(z_scores)
        
        # Manter apenas linhas onde TODAS as features têm z < k
        mask_keep = (z_scores < k).all(axis=1)
        
        # Se a limpeza for remover TUDO (erro comum), mantemos o original
        if np.sum(mask_keep) == 0:
            X_clean_list.append(X_c)
            y_clean_list.append(y[mask_c])
        else:
            X_clean_list.append(X_c[mask_keep])
            y_clean_list.append(y[mask_c][mask_keep])
        
    if not X_clean_list: return X, y
    
    X_clean = np.vstack(X_clean_list)
    y_clean = np.concatenate(y_clean_list)
    
    return X_clean, y_clean

def augment_with_smote_strategy(X, y, strategy_ratio=0.25):
    """
    Adiciona 25% da diferença para a classe maioritária usando SMOTE.
    Protegido contra arrays vazios.
    """
    # --- PROTEÇÃO CONTRA ERRO ---
    if X is None or len(X) == 0 or len(y) == 0:
        return X, y
        
    unique, counts = np.unique(y, return_counts=True)
    
    # Se só houver 1 classe ou nenhuma, não dá para balancear/calcular max
    if len(unique) < 2:
        return X, y
    
    max_count = np.max(counts) # Agora é seguro chamar max
    
    X_aug_list = [X]
    y_aug_list = [y]
    
    generated_count = 0
    
    for cls, count in zip(unique, counts):
        gap = max_count - count
        n_to_generate = int(gap * strategy_ratio)
        
        # Só gera se valer a pena (>1 amostra) e se houver dados suficientes para vizinhos
        if n_to_generate > 1:
            X_class = X[y == cls]
            
            # Precisamos de pelo menos 2 amostras para calcular vizinhos
            if len(X_class) < 2: continue

            # Tenta gerar
            try:
                # Reutiliza a função do Ex 1.2
                # Ajustamos k_neighbors dinamicamente para evitar erro
                k_neigh = min(5, len(X_class) - 1)
                if k_neigh < 1: k_neigh = 1
                
                synth = generate_smote_samples(X_class, k_samples=n_to_generate, k_neighbors=k_neigh)
                
                if synth is not None and len(synth) > 0:
                    X_aug_list.append(synth)
                    y_synth = np.full(len(synth), cls)
                    y_aug_list.append(y_synth)
                    generated_count += 1
            except Exception:
                # Se o SMOTE falhar numa classe específica, ignora e continua
                continue
    
    if generated_count > 0:
        return np.vstack(X_aug_list), np.concatenate(y_aug_list)
        
    return X, y    

# =============================================================================
# MODELO HIERÁRQUICO (CLASSE WRAPPER)
# =============================================================================

class HierarchicalClassifier:
    """
    Classificador 'Two-Stage' compatível com sklearn.
    - Estágio 1: Decide se é Estático ou Dinâmico.
    - Estágio 2: Usa especialistas para decidir a classe final.
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        # Generalista: Random Forest (Robusto para decisões gerais)
        self.clf_general = RandomForestClassifier(n_estimators=100, random_state=random_state)
        
        # Especialista Estático (1,2,3): Gradient Boosting (Bom para detalhes finos Sit vs SitTalk)
        self.clf_static = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=random_state)
        
        # Especialista Dinâmico (4,5,6,7): Random Forest (Bom para variância alta de movimento)
        self.clf_dynamic = RandomForestClassifier(n_estimators=100, random_state=random_state)
        
        self.static_classes = [1, 2, 3]
        self.dynamic_classes = [4, 5, 6, 7]

    def _get_group(self, y):
        # 0 = Estático, 1 = Dinâmico
        groups = np.zeros_like(y)
        groups[np.isin(y, self.dynamic_classes)] = 1
        return groups

    def fit(self, X, y):
        # 1. Treinar Generalista
        y_groups = self._get_group(y)
        self.clf_general.fit(X, y_groups)
        
        # 2. Treinar Especialista Estático
        mask_static = np.isin(y, self.static_classes)
        if np.sum(mask_static) > 0:
            self.clf_static.fit(X[mask_static], y[mask_static])
            
        # 3. Treinar Especialista Dinâmico
        mask_dynamic = np.isin(y, self.dynamic_classes)
        if np.sum(mask_dynamic) > 0:
            self.clf_dynamic.fit(X[mask_dynamic], y[mask_dynamic])
        return self

    def predict(self, X):
        # 1. Previsão do Grupo
        group_preds = self.clf_general.predict(X)
        final_preds = np.zeros_like(group_preds)
        
        # 2. Previsão Especializada
        # Onde é estático (0)
        mask_static = (group_preds == 0)
        if np.any(mask_static):
            final_preds[mask_static] = self.clf_static.predict(X[mask_static])
            
        # Onde é dinâmico (1)
        mask_dynamic = (group_preds == 1)
        if np.any(mask_dynamic):
            final_preds[mask_dynamic] = self.clf_dynamic.predict(X[mask_dynamic])
            
        return final_preds

# =============================================================================
# FUNÇÃO DE COMPARAÇÃO DE 4 MODELOS
# =============================================================================

# =============================================================================
# COMPARAÇÃO ESTATÍSTICA DE 4 MODELOS (ATUALIZADA)
# =============================================================================

def compare_4_models_statistical(X_raw, y_raw, p_raw, n_repeats=5):
    """
    Roda n_repeats iterações de treino/teste para 4 modelos.
    Imprime tabela detalhada e gera Box Plot comparativo.
    """
    print(f"\n>>> A iniciar Comparação de 4 Modelos ({n_repeats} iterações)...")
    
    # 1. Definir os Candidatos
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, class_weight='balanced', n_jobs=-1, random_state=42),
        "SVM (RBF)": SVC(kernel='rbf', C=1.0, class_weight='balanced', random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
        "Hierárquico (Otimizado)": HierarchicalClassifier(random_state=42)
    }
    
    # Dicionário para guardar listas de scores
    results = {name: [] for name in models.keys()}
    
    # 2. Loop de Avaliação
    for i in range(n_repeats):
        seed = 42 + i
        
        # Split Consistente (Mesma seed para todos na mesma iteração)
        split_res = split_between_subjects(X_raw, y_raw, p_raw, seed=seed)
        if split_res is None: continue
        X_tr, y_tr, X_val, y_val, X_te, y_te = split_res
        
        # Pipeline Normalização
        scaler = StandardScaler()
        X_tr_norm = scaler.fit_transform(X_tr)
        X_val_norm = scaler.transform(X_val)
        X_te_norm = scaler.transform(X_te)
        
        # Juntar Treino + Validação
        X_full = np.vstack((X_tr_norm, X_val_norm))
        y_full = np.concatenate((y_tr, y_val))
        
        # Treinar e Testar cada modelo
        for name, clf in models.items():
            clf.fit(X_full, y_full)
            y_pred = clf.predict(X_te_norm)
            acc = accuracy_score(y_te, y_pred)
            results[name].append(acc)
            
        # Feedback de progresso
        print(f"\r   -> Iteração {i+1}/{n_repeats} concluída.", end="")

    # 3. Imprimir Tabela de Resultados Detalhada
    print("\n\n" + "="*80)
    print(f"=== RESULTADOS DETALHADOS POR ITERAÇÃO (Accuracy) ===")
    print("="*80)
    
    # Cabeçalho da Tabela
    header = f"{'Iter':<5} | " + " | ".join([f"{name:<23}" for name in models.keys()])
    print(header)
    print("-" * len(header))
    
    # Linhas da Tabela
    for i in range(len(results["Random Forest"])): # Usa o tamanho da lista do primeiro modelo
        row_str = f"{i+1:<5} | "
        row_str += " | ".join([f"{results[name][i]:.4f}" for name in models.keys()])
        print(row_str)
    print("-" * len(header))
    
    # 4. Análise Estatística (Médias e Testes)
    print("\n" + "="*60)
    print("=== ANÁLISE ESTATÍSTICA FINAL ===")
    print("="*60)
    
    means = {m: np.mean(s) for m, s in results.items()}
    stds = {m: np.std(s) for m, s in results.items()}
    best_model = max(means, key=means.get)
    best_scores = results[best_model]
    
    print(f"🏆 VENCEDOR: {best_model}")
    print(f"   Média: {means[best_model]:.4f} (+/- {stds[best_model]:.4f})")
    print("-" * 60)
    
    # Testes de Hipótese (Best vs Others)
    for name, scores in results.items():
        if name == best_model: continue
        
        t_stat, p_val = ttest_rel(best_scores, scores)
        diff = means[best_model] - means[name]
        signif = "SIM ✅" if p_val < 0.05 else "NÃO ❌"
        
        print(f"vs {name:<23} | Dif: +{diff:.4f} | p={p_val:.4f} | Sig? {signif}")

    # 5. Visualização (Box Plot Comparativo)
    print("\n-> A gerar Box Plot comparativo...")
    plt.figure(figsize=(12, 7))
    
    # Boxplot
    plt.boxplot(results.values(), labels=results.keys(), patch_artist=True,
                boxprops=dict(facecolor="lightgreen", alpha=0.6),
                medianprops=dict(color="black"))
    
    # Adicionar pontos individuais (jitter) para ver a dispersão real
    for i, (name, scores) in enumerate(results.items()):
        y = scores
        x = np.random.normal(i + 1, 0.04, size=len(y)) # Adiciona ruído no eixo X
        plt.plot(x, y, 'r.', alpha=0.5)
        
    plt.title(f"Comparação de Performance (4 Modelos) - {n_repeats} Repetições")
    plt.ylabel("Accuracy (Teste)")
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    path_img = os.path.join(outdir, "comparacao_4_modelos.png")
    savefig(path_img)
    print(f"Gráfico guardado em: {path_img}")

    
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
    # EXERCÍCIO 4 e 5: AVALIAÇÃO COM REPETIÇÕES & TESTES
    # =================================================================
    print("\n=== 5. Evaluation Loop (Tuning, Retrain & Stats) ===")
    
    # --- ALTERAÇÃO: 20 Repetições ---
    N_REPEATS = 1
    K_VALUES_LIST = [1, 3, 5, 7, 9, 11, 13, 15]
    
    # Dicionário para acumular resultados: {'FEATURES-A': [0.62, 0.61...], ...}
    results_history = {}

    for i in range(N_REPEATS):
        current_seed = 42 + i
        print(f"\n>> Iteração {i+1}/{N_REPEATS} (Seed {current_seed})...")
        
        for ds_name in ready_data:
            # Recuperar dados brutos para fazer novo split
            if ds_name == "FEATURES": original_data = data_features
            else: original_data = data_embeds
            
            X = original_data[:, IDX_FEATS:]
            y = original_data[:, IDX_LABEL]
            p = original_data[:, IDX_PART]
            
            # 1. Split Variável (Seed muda a cada iteração)
            split_res = split_between_subjects(X, y, p, seed=current_seed)
            if split_res is None: continue
            X_tr, y_tr, X_val, y_val, X_te, y_te = split_res
            
            # 2. Pipeline
            scenarios = process_pipeline_scenarios(X_tr, X_val, X_te, y_tr)
            
            for sc_name, sc_data in scenarios.items():
                if sc_data is None: continue
                
                X_train_s, X_val_s, X_test_s = sc_data
                
                # 5.1 Tuning & Retrain
                # Nota: Certifique-se que está a usar a versão da função que devolve 4 valores!
                acc_test, best_k, _, _ = tune_and_retrain(
                    X_train_s, y_tr, X_val_s, y_val, X_test_s, y_te, 
                    k_values=K_VALUES_LIST
                )
                
                # Guardar histórico
                key = f"{ds_name}-{sc_name}"
                if key not in results_history: results_history[key] = []
                results_history[key].append(acc_test)
                
                print(f"   [{key}] k={best_k}, Acc={acc_test:.4f}")

    # --- Gráfico da Distribuição (NOVO) ---
    if results_history:
        plot_results_distribution(results_history, outdir)

    # 5.3 Testes de Hipótese
    perform_hypothesis_testing(results_history)

    print("\nPipeline Final Concluída!")


    # =================================================================
    # EXERCÍCIO 6: DEPLOYMENT (Teste com 'teste.csv')
    # =================================================================
    print("\n=== 6. Deployment (Aplicação Real) ===")
    
    # --- Configuração do Modelo de Produção ---
    # Escolhemos o modelo FEATURES (Cenário A ou B) pois teve melhor performance no Ex 5.
    # k=15 foi um valor consistente nos seus resultados.
    BEST_K_PROD = 15
    USAR_PCA = False # Coloque True se preferir o modelo com PCA
    
    if data_features is not None:
        # Preparar dados totais para treino
        X_all = data_features[:, IDX_FEATS:]
        y_all = data_features[:, IDX_LABEL]
        
        # 1. Treinar o 'Deployer' (Cria o cérebro do modelo)
        deployer = train_deployable_model(X_all, y_all, best_k=BEST_K_PROD, use_pca=USAR_PCA)
        
        # 2. Carregar o ficheiro de teste 'teste.csv'
        TEST_FILE = "teste.csv"
        print(f"[Deploy] A procurar ficheiro '{TEST_FILE}' na pasta...")
        
        if os.path.exists(TEST_FILE):
            try:
                # Carregar ignorando cabeçalhos se existirem
                raw_input = np.loadtxt(TEST_FILE, delimiter=',')
                
                # Validação básica
                if raw_input.ndim == 2 and raw_input.shape[1] >= 9:
                    # Garantir que usamos apenas as primeiras 9 colunas (Acc, Gyr, Mag)
                    # Caso o ficheiro tenha mais (ex: timestamps)
                    input_data = raw_input[:, :9]
                    
                    print(f"[Deploy] Ficheiro carregado com sucesso: {input_data.shape}")
                    print("[Deploy] A processar janelas e extrair features...")
                    
                    # 3. Previsão
                    predicao = deployer.predict(input_data)
                    
                    print(f"\n>>>O MODELO PREVIU A ATIVIDADE: {predicao}")
                    
                else:
                    print(f"Erro: '{TEST_FILE}' tem formato inválido. Esperado: (N, 9+). Encontrado: {raw_input.shape}")
            except Exception as e:
                print(f"Erro ao ler '{TEST_FILE}': {e}")
        else:
            print(f"Aviso: Ficheiro '{TEST_FILE}' não encontrado.")
            print("Crie um ficheiro CSV com 256 linhas e 9 colunas para testar esta funcionalidade.")
            
    else:
        print("Erro: Não existem dados de Features para treinar o modelo de produção.")

    # =================================================================
    # EXERCÍCIO EXTRA 1: Teste de Subset (Sem confusões)
    # =================================================================
    print("\n=== EXTRA 1: Avaliação com Subset (Sem Atividades 3 e 5) ===")
    
    # Definir atividades a manter
    KEEP_LABELS = [1, 2, 4, 6, 7]
    
    if data_features is not None:
        print(f"Filtrando apenas atividades: {KEEP_LABELS}")
        
        # 1. Filtrar
        mask_subset = np.isin(data_features[:, IDX_LABEL], KEEP_LABELS)
        data_sub = data_features[mask_subset]
        
        X_sub = data_sub[:, IDX_FEATS:]
        y_sub = data_sub[:, IDX_LABEL]
        p_sub = data_sub[:, IDX_PART]
        
        # 2. Split (Seed fixa para teste rápido)
        split_res = split_between_subjects(X_sub, y_sub, p_sub, seed=42)
        
        if split_res is not None:
            X_tr, y_tr, X_val, y_val, X_te, y_te = split_res
            
            # 3. Pipeline (Normalização)
            scaler = StandardScaler()
            X_tr_norm = scaler.fit_transform(X_tr)
            X_val_norm = scaler.transform(X_val)
            
            # 4. Treino e Avaliação (Usando k=15 que foi o melhor antes)
            print("Treinando k-NN (k=15) no subset...")
            clf = KNeighborsClassifier(n_neighbors=15)
            clf.fit(X_tr_norm, y_tr)
            
            y_pred = clf.predict(X_val_norm)
            
            calculate_metrics(y_val, y_pred, set_name="SUBSET 1-2-4-6-7")

        # =================================================================
        # COMPARAÇÃO FINAL: 4 MODELOS
        # =================================================================
        print("\n=== COMPARAÇÃO FINAL DE 4 MODELOS (RF, SVM, GB, Hierárquico) ===")
        
        if "FEATURES" in ready_data:
            # Dados Base
            X_raw = data_features[:, IDX_FEATS:]
            y_raw = data_features[:, IDX_LABEL]
            p_raw = data_features[:, IDX_PART]
            
            # Executar a função de comparação (20 iterações)
            compare_4_models_statistical(X_raw, y_raw, p_raw, n_repeats=5)
                
        else:
            print("Erro: Dados FEATURES indisponíveis.")
            
    
if __name__ == "__main__":
    main()