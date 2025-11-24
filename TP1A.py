# Rodrigo Martins Ceia nº2023222356
# Gabriel Alexandre Sabino Costeira nº 2023222421
import os
import csv
import warnings
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from scipy import stats, signal
from scipy.stats import kstest, zscore, skew, kurtosis, iqr, entropy

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ------------------------------
# Utilitários de saída (guardar figuras/CSV)
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
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if header: writer.writerow(header)
        writer.writerows(rows)

_plot_counter = 0
outdir = ensure_outputs_dir("outputs")

def next_plot_id(prefix="plot"):
    global _plot_counter
    _plot_counter += 1
    return f"{prefix}_{_plot_counter:03d}.png"



# ------------------------------
# Exercício 2
# ------------------------------

def carregar_dados(part_id, pasta_base="FORTH_TRACE_DATASET", devices=(1,2,3,4,5)):
    """Carrega os ficheiros de um participante para um único numpy array.
    Cada linha assume o formato: [DeviceID, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, mag_x, mag_y, mag_z, ts, activity]
    Retorna: array shape (N, 12)
    """
    rows = []
    for dev in devices:
        path = os.path.join(pasta_base, f"part{part_id}", f"part{part_id}dev{dev}.csv")
        if not os.path.isfile(path):
            continue
        with open(path, "r") as f:
            reader = csv.reader(f)
            for r in reader:
                if not r:
                    continue
                try:
                    vals = [float(x) for x in r]
                    rows.append(vals)
                except Exception:
                    continue
    if len(rows) == 0:
        raise FileNotFoundError(f"Nenhum ficheiro encontrado para part{part_id} em {pasta_base}")
    return np.array(rows)

# ------------------------------
# Exercício 3
# ------------------------------

def calcular_modulo(vetor_x, vetor_y, vetor_z):
    """Calcula o módulo do vetor tridimensional."""
    return np.sqrt(vetor_x**2 + vetor_y**2 + vetor_z**2)

# ------------------------------
# Exercício 3.1
# ------------------------------
def boxplot_por_atividade(modulo_vals, labels, titulo="Boxplot por atividade"):
    """Plota boxplot dos módulos por atividade."""
    modulo_vals = np.array(modulo_vals)
    labels = np.array(labels)

    atividades_unicas = np.unique(labels)
    atividades_unicas.sort()
    dados_box = [modulo_vals[labels == a] for a in atividades_unicas]

    plt.figure(figsize=(12, 6))
    plt.boxplot(dados_box, tick_labels=atividades_unicas)
    plt.title(titulo)
    plt.xlabel("Atividade (label)")
    plt.ylabel("Módulo")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(os.path.join(outdir, next_plot_id()))
    plt.close()

# ------------------------------
# Exercício 3.2
# ------------------------------
def identificar_outliers_tukey(amostras):
    """Identifica outliers usando o método de Tukey (IQR).
       Retorna um array booleano com True nos outliers."""
    amostras = np.asarray(amostras)

    Q1 = np.percentile(amostras, 25)
    Q3 = np.percentile(amostras, 75)
    IQR = Q3 - Q1

    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR

    outliers_mask = (amostras < limite_inferior) | (amostras > limite_superior)

    return outliers_mask

def densidade_outliers_tukey(modulos, labels):
    """Calcula densidade de outliers (método de Tukey) por atividade."""
    atividades_unicas = np.unique(labels)
    densidades = {}

    for a in atividades_unicas:
        dados_a = modulos[labels == a]

        outliers_mask = identificar_outliers_tukey(dados_a)

        n_outliers = np.sum(outliers_mask)
        n_total = len(dados_a)

        densidades[a] = (n_outliers / n_total) * 100

    return densidades


def print_densidade_legivel(densidades, sensor_nome="Sensor"):
    print(f"\nDensidade de outliers - {sensor_nome}:")
    print(f"{'Atividade':>10} | {'Densidade (%)':>15}")
    print("-" * 30)
    for atividade in sorted(densidades.keys()):
        valor = float(densidades[atividade])
        print(f"{int(atividade):>10} | {valor:>15.3f}")

# ------------------------------
# Exercício 3.3
# ------------------------------
def identificar_outliers_zscore_k(amostras, k):
    """Identifica índices e valores outliers com base no Z-Score e limiar k."""
    amostras = np.array(amostras)
    z_scores = np.abs(zscore(amostras))
    indices_outliers = np.where(z_scores > k)[0]
    valores_outliers = amostras[indices_outliers]
    
    return indices_outliers, valores_outliers

def densidade_outliers_zscore(modulos, labels, k):
    """Calcula densidade de outliers usando Z-Score com limiar k por atividade."""
    atividades_unicas = np.unique(labels)
    densidades = {}
    for a in atividades_unicas:
        dados_a = modulos[labels == a]
        indices_outliers, _ = identificar_outliers_zscore_k(dados_a, k)
        n_outliers = len(indices_outliers)
        n_total = len(dados_a)
        densidades[a] = (n_outliers / n_total) * 100
    return densidades
# ------------------------------
# Exercício 3.4
# ------------------------------


def plotar_outliers(amostras, k, titulo, labels):
    """Plota as amostras com outliers (vermelho) e normais (azul) para um dado k, por atividade."""
    indices_outliers, _ = identificar_outliers_zscore_k(amostras, k)
    
    mask_outliers = np.zeros(len(amostras), dtype=bool)
    mask_outliers[indices_outliers] = True
    
    plt.figure(figsize=(12, 6))
    
    plt.scatter(labels[~mask_outliers], amostras[~mask_outliers], 
                color='blue', label='Normal', alpha=0.5, s=10)
    
    plt.scatter(labels[mask_outliers], amostras[mask_outliers], 
                color='red', label='Outliers', alpha=0.7, s=20)
    
    plt.title(f"{titulo} (k={k})")
    plt.xlabel("Atividade (label)")
    plt.ylabel("Módulo")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(os.path.join(outdir, next_plot_id()))
    plt.close()


def visualizar_outliers_multik(amostras, nome_sensor, labels):
    """Gera plots para k = 3, 3.5 e 4."""
    for k in [3, 3.5, 4]:
        plotar_outliers(amostras, k, f"{nome_sensor} - Outliers por Z-Score", labels)

# ------------------------------
# Exercício 3.6
# ------------------------------


def kmeans(dados, n_clusters, max_iter=100, tol=1e-4):
    """Implementa o algoritmo k-means para n clusters."""
    dados = np.array(dados).reshape(-1, 1)
    np.random.seed(0)
    indices_iniciais = np.random.choice(len(dados), n_clusters, replace=False)
    centroides = dados[indices_iniciais]

    for _ in range(max_iter):
        distancias = np.abs(dados - centroides.T)
        atribuicoes = np.argmin(distancias, axis=1)
        novos_centroides = np.array([dados[atribuicoes == k].mean() if np.any(atribuicoes == k) else centroides[k] for k in range(n_clusters)])
        if np.all(np.abs(novos_centroides - centroides) < tol):
            break
        centroides = novos_centroides
    return atribuicoes, centroides

def identificar_outliers_kmeans(dados, atribuicoes, centroides, limiar=2.5):
    """
    Identifica outliers com base na distância ao centróide.
    Funciona tanto para dados 1D (escalares) quanto multidimensionais (vetores).
    """
    dados = np.array(dados)

    if dados.ndim == 1:
        distancias = np.abs(dados - centroides[atribuicoes])
    else:
        distancias = np.linalg.norm(dados - centroides[atribuicoes], axis=1)

    media = np.mean(distancias)
    desvio = np.std(distancias)
    limiar_dist = media + limiar * desvio

    indices_outliers = np.where(distancias > limiar_dist)[0]

    print(f"Número de outliers identificados (identificar_outliers_kmeans): {len(indices_outliers)}")
    return indices_outliers, distancias

def densidade_outliers_kmeans(dados, labels, n_clusters, limiar=2.5):
    """Calcula densidade de outliers usando K-Means por atividade."""
    atividades_unicas = np.unique(labels)
    densidades = {}
    for a in atividades_unicas:
        mask_atividade = labels == a
        dados_a = dados[mask_atividade]
        
        if len(dados_a) < n_clusters:
            densidades[a] = 0.0
            continue
        
        atribuicoes, centroides = kmeans_multivariado(dados_a, n_clusters)
        indices_outliers, _ = identificar_outliers_kmeans(dados_a, atribuicoes, centroides, limiar)
        n_outliers = len(indices_outliers)
        n_total = len(dados_a)
        densidades[a] = (n_outliers / n_total) * 100
        print("Número de outliers (densidade_outliers_kmeans): ", n_outliers)
        print("número total:", n_total)
    return densidades

def plotar_clusters(dados, atribuicoes, centroides, titulo, labels, outliers=None):
    """Plota os dados agrupados, seus centroides e outliers (em vermelho)."""
    dados = np.array(dados)
    labels = np.array(labels)

    plt.figure(figsize=(12, 6))

    # Plotar clusters
    for k in range(len(centroides)):
        plt.scatter(
            labels[atribuicoes == k],
            dados[atribuicoes == k],
            label=f"Cluster {k+1}",
            alpha=0.6,
            s=10
        )

    # Plotar outliers (se existirem)
    if outliers is not None and len(outliers) > 0:
        plt.scatter(
            labels[outliers],
            dados[outliers],
            color='red',
            s=30,
            label='Outliers'
        )

    # Plotar centroides
    plt.scatter(
        np.repeat(np.mean(labels), len(centroides)),
        centroides,
        color='black',
        marker='x',
        s=100,
        label='Centroides'
    )

    plt.title(titulo)
    plt.xlabel("Atividade (label)")
    plt.ylabel("Módulo")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(os.path.join(outdir, next_plot_id()))
    plt.close()



# ------------------------------
# Exercício 3.7
# ------------------------------


def kmeans_multivariado(dados, n_clusters, max_iter=100, tol=1e-4):
    """Implementa o algoritmo k-means para dados multivariados."""
    dados = np.array(dados)
    np.random.seed(0)
    indices_iniciais = np.random.choice(len(dados), n_clusters, replace=False)
    centroides = dados[indices_iniciais]

    for _ in range(max_iter):
        distancias = np.linalg.norm(dados[:, np.newaxis] - centroides, axis=2)
        atribuicoes = np.argmin(distancias, axis=1)
        novos_centroides = np.array([
            dados[atribuicoes == k].mean(axis=0) if np.any(atribuicoes == k) else centroides[k]
            for k in range(n_clusters)
        ])
        if np.all(np.linalg.norm(novos_centroides - centroides, axis=1) < tol):
            break
        centroides = novos_centroides
    return atribuicoes, centroides


def plotar_clusters_3d(dados, atribuicoes, centroides, indices_outliers, titulo):
    """Plota clusters 3D e as projeções 2D (xOy, xOz, yOz), destacando outliers."""
    dados = np.array(dados)
    centroides = np.array(centroides)

    # Cria figura com 4 subplots
    fig = plt.figure(figsize=(14, 10))

    # ---------------- Gráfico 3D ----------------
    ax3d = fig.add_subplot(221, projection='3d')
    for k in range(len(centroides)):
        cluster_points = dados[atribuicoes == k]
        ax3d.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
                     label=f"Cluster {k+1}", s=15)

    
    if len(indices_outliers) > 0:
        outliers = dados[indices_outliers]
        ax3d.scatter(outliers[:, 0], outliers[:, 1], outliers[:, 2],
                     color='red', label='Outliers', s=30)

    
    ax3d.scatter(centroides[:, 0], centroides[:, 1], centroides[:, 2],
                 color='black', marker='x', s=80, label='Centroides')

    ax3d.set_title(f"{titulo} - 3D")
    ax3d.set_xlabel("Eixo X")
    ax3d.set_ylabel("Eixo Y")
    ax3d.set_zlabel("Eixo Z")
    ax3d.legend()

    # ---------------- Projeção xOy ----------------
    ax_xy = fig.add_subplot(222)
    for k in range(len(centroides)):
        cluster_points = dados[atribuicoes == k]
        ax_xy.scatter(cluster_points[:, 0], cluster_points[:, 1], s=10)
    if len(indices_outliers) > 0:
        ax_xy.scatter(outliers[:, 0], outliers[:, 1], color='red', s=20, label='Outliers')
    ax_xy.scatter(centroides[:, 0], centroides[:, 1], color='black', marker='x', s=80)
    ax_xy.set_xlabel("Eixo X")
    ax_xy.set_ylabel("Eixo Y")
    ax_xy.set_title("Projeção xOy")

    # ---------------- Projeção xOz ----------------
    ax_xz = fig.add_subplot(223)
    for k in range(len(centroides)):
        cluster_points = dados[atribuicoes == k]
        ax_xz.scatter(cluster_points[:, 0], cluster_points[:, 2], s=10)
    if len(indices_outliers) > 0:
        ax_xz.scatter(outliers[:, 0], outliers[:, 2], color='red', s=20)
    ax_xz.scatter(centroides[:, 0], centroides[:, 2], color='black', marker='x', s=80)
    ax_xz.set_xlabel("Eixo X")
    ax_xz.set_ylabel("Eixo Z")
    ax_xz.set_title("Projeção xOz")

    # ---------------- Projeção yOz ----------------
    ax_yz = fig.add_subplot(224)
    for k in range(len(centroides)):
        cluster_points = dados[atribuicoes == k]
        ax_yz.scatter(cluster_points[:, 1], cluster_points[:, 2], s=10)
    if len(indices_outliers) > 0:
        ax_yz.scatter(outliers[:, 1], outliers[:, 2], color='red', s=20)
    ax_yz.scatter(centroides[:, 1], centroides[:, 2], color='black', marker='x', s=80)
    ax_yz.set_xlabel("Eixo Y")
    ax_yz.set_ylabel("Eixo Z")
    ax_yz.set_title("Projeção yOz")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, next_plot_id()))
    plt.close()
""""""
# -------------
# Exercício 4.1
# -------------

# Significância estatística e normalidade por atividade
# Variáveis usadas: módulos (acel, giro, mag) – já calculados.

def ks_normality_by_activity(values, labels):
    """
    Teste de normalidade por atividade via Kolmogorov–Smirnov.
    Estratégia: z-score por atividade e K-S vs N(0,1).
    Retorna dict: atividade -> (statistic, pvalue, n)
    """
    res = {}
    for a in np.unique(labels):
        v = values[labels == a]
        v = v[np.isfinite(v)]
        if len(v) < 5:
            res[int(a)] = (np.nan, np.nan, len(v))
            continue
        z = (v - np.mean(v)) / (np.std(v) + 1e-12)
        stat, p = kstest(z, 'norm')
        res[int(a)] = (float(stat), float(p), int(len(v)))
    return res

def test_means_across_activities(values, labels, alpha=0.05):
    """
    Compara médias entre atividades.
    - Se TODAS as atividades ~ normal (KS p>0.05) e variâncias ~ semelhantes
      (Levene p>0.05), usa ANOVA (one-way).
    - Caso contrário, usa Kruskal–Wallis.
    Retorna dict com método e p-valor.
    """
    groups = []
    normals = []
    for a in np.unique(labels):
        v = values[labels == a]
        v = v[np.isfinite(v)]
        if len(v) >= 5:
            groups.append(v)
            z = (v - np.mean(v)) / (np.std(v) + 1e-12)
            _, pks = kstest(z, 'norm')
            normals.append(pks > 0.05)

    method = ""
    pvalue = np.nan
    if len(groups) >= 2:
        # Levene para homogeneidade de variâncias
        try:
            stat_lev, p_lev = stats.levene(*groups)
        except Exception:
            p_lev = 0.0

        if all(normals) and (p_lev > 0.05):
            method = "ANOVA (one-way)"
            _, pvalue = stats.f_oneway(*groups)
        else:
            method = "Kruskal–Wallis"
            _, pvalue = stats.kruskal(*groups)
    return {"method": method, "pvalue": float(pvalue)}

# -------------
# Exercício 4.2
# -------------

# Extração de features temporais e espectrais

def time_features(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "mean": np.nan, "std": np.nan, "var": np.nan, "rms": np.nan,
            "median": np.nan, "iqr": np.nan, "min": np.nan, "max": np.nan,
            "skew": np.nan, "kurt": np.nan, "zcr": np.nan
        }
    diff_sign = np.sign(x[1:]) != np.sign(x[:-1])
    zcr = float(np.sum(diff_sign)) / (len(x) - 1 + 1e-12)

    feats = {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "var": float(np.var(x)),
        "rms": float(np.sqrt(np.mean(x**2))),
        "median": float(np.median(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "skew": float(skew(x, bias=False) if len(x) > 2 else 0.0),
        "kurt": float(kurtosis(x, bias=False) if len(x) > 3 else 0.0),
        "zcr": zcr
    }
    return feats

def spectral_features(x, fs):
    """
    Features espectrais simples a partir de |FFT|:
    - centroid, bandwidth, peak_freq, spectral_entropy, spectral_flatness
    - + spectral_rolloff_85, spectral_crest, spectral_contrast, spectral_energy, spectral_spread
    """
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {k: np.nan for k in [
            "spec_centroid","spec_bandwidth","peak_freq","spec_entropy",
            "spec_flatness","spec_rolloff_85","spec_crest","spec_contrast",
            "spec_energy","spec_spread"
        ]}

    # Janela Hanning + FFT
    w = np.hanning(len(x))
    X = np.fft.rfft((x - np.mean(x)) * w)
    mag = np.abs(X)
    freqs = np.fft.rfftfreq(len(x), d=1.0/fs)
    psd = (mag**2)
    psd_sum = np.sum(psd) + 1e-12

    # --- Features base ---
    centroid = np.sum(freqs * psd) / psd_sum
    var = np.sum(((freqs - centroid)**2) * psd) / psd_sum
    bandwidth = np.sqrt(max(var, 0.0))

    peak_idx = int(np.argmax(psd))
    peak_freq = float(freqs[peak_idx])

    p = psd / psd_sum
    spec_entropy = float(entropy(p))

    geom_mean = np.exp(np.mean(np.log(psd + 1e-12)))
    arith_mean = np.mean(psd) + 1e-12
    spec_flatness = float(geom_mean / arith_mean)

    # --- Novas features ---
    cumulative_psd = np.cumsum(psd)
    rolloff_threshold = 0.85 * psd_sum
    rolloff_idx = np.where(cumulative_psd >= rolloff_threshold)[0][0]
    spec_rolloff_85 = float(freqs[rolloff_idx])

    spec_crest = float(np.max(mag) / (np.mean(mag) + 1e-12))

    n_bands = 6
    band_edges = np.linspace(0, len(psd), n_bands + 1, dtype=int)
    band_max = [np.max(psd[band_edges[i]:band_edges[i+1]]) for i in range(n_bands)]
    band_min = [np.min(psd[band_edges[i]:band_edges[i+1]]) for i in range(n_bands)]
    spec_contrast = float(np.mean(band_max) - np.mean(band_min))

    # 4. Spectral energy (total)
    spec_energy = float(psd_sum)

    # 5. Spectral spread
    spec_spread = float(np.sqrt(np.sum(((freqs - centroid)**2) * psd) / psd_sum))

    return {
        "spec_centroid": float(centroid),
        "spec_bandwidth": float(bandwidth),
        "peak_freq": float(peak_freq),
        "spec_entropy": float(spec_entropy),
        "spec_flatness": float(spec_flatness),
        "spec_rolloff_85": float(spec_rolloff_85),
        "spec_crest": float(spec_crest),
        "spec_contrast": float(spec_contrast),
        "spec_energy": float(spec_energy),
        "spec_spread": float(spec_spread),
    }


def extract_window_features(x, fs):
    f_time = time_features(x)
    f_spec = spectral_features(x, fs=fs)
    feats = {}
    feats.update(f_time)
    feats.update(f_spec)
    return feats

def sliding_window_features(signal_array, labels_array, participant_array, device_array, fs):
    """
    Extrai features, garantindo que a janela pertence ao mesmo:
    1. Label (Atividade)
    2. Participante
    3. Device (Sensor) <--- NOVO FILTRO CRÍTICO
    """
    win_size = int(5 * fs)
    step = int(win_size / 2)

    X_list = []
    y_list = []
    p_list = [] 
    # (Opcional: pode querer guardar o device ID também, mas não é estritamente necessário para o output)

    for start in range(0, len(signal_array) - win_size + 1, step):
        end = start + win_size
        
        xw = signal_array[start:end]
        lw = labels_array[start:end]
        pw = participant_array[start:end]
        dw = device_array[start:end] # <--- Janela de Devices

        # Verifica unicidade de Label, Participante E DEVICE
        unique_labels = np.unique(lw)
        unique_parts = np.unique(pw)
        unique_devs = np.unique(dw) # <--- Verifica se há apenas 1 device nesta janela

        # Se houver mais que 1 device, significa que estamos na transição entre sensores.
        # Devemos descartar.
        if len(unique_labels) != 1 or len(unique_parts) != 1 or len(unique_devs) != 1:
            continue

        feats = extract_window_features(xw, fs=fs)
        X_list.append(list(feats.values()))
        y_list.append(unique_labels[0])
        p_list.append(unique_parts[0])

    if not X_list:
        return np.empty((0, 0)), np.array([]), np.array([]), []

    feature_names = list(extract_window_features(signal_array[:win_size], fs=fs).keys())
    
    return np.array(X_list, dtype=float), np.array(y_list, dtype=float), np.array(p_list, dtype=int), feature_names

# -------------
# Exercício 4.3
# -------------

#  PCA
def run_pca(X, feature_names=None, var_norm=True, top_k=5):
    """
    Executa PCA sobre o feature set X e imprime as variáveis com maior peso
    em cada componente principal.

    Args:
        X (array): matriz de features [n_samples, n_features]
        feature_names (list): nomes das features
        var_norm (bool): normaliza as variáveis antes do PCA
        top_k (int): número de variáveis mais influentes a mostrar por componente
    """
    scaler = StandardScaler(with_mean=True, with_std=True) if var_norm else None
    Xn = scaler.fit_transform(X) if scaler is not None else X
    pca = PCA()
    Z = pca.fit_transform(Xn)

   
    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(X.shape[1])]

    print("\n========== PCA – Variáveis com maior peso por componente ==========")
    for i, comp in enumerate(pca.components_):
        top_indices = np.argsort(np.abs(comp))[::-1][:top_k]
        print(f"\nComponente {i+1} (variância explicada: {pca.explained_variance_ratio_[i]*100:.2f}%):")
        for idx in top_indices:
            peso = comp[idx]
            print(f"  {feature_names[idx]:<25}  loading = {peso:+.4f}")
        if i == 9:
            break
    print("====================================================================")

    return {"scaler": scaler, "pca": pca, "Z": Z}


# --------------
# Exercício 4.4
# --------------

# Variância explicada + nº de dimensões para 75%

def num_components_for_variance(pca, threshold=0.75):
    cum = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cum, threshold) + 1)
    return k, cum

def plot_variance_explained(pca, out_path):
    cum = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(8,4))
    plt.plot(np.arange(1, len(cum)+1), cum, marker='o')
    plt.axhline(0.75, linestyle='--')
    plt.xlabel("Número de componentes")
    plt.ylabel("Variância explicada acumulada")
    plt.title("PCA – Variância explicada acumulada")
    plt.grid(True, linestyle="--", alpha=0.6)
    savefig(out_path)

# -------------
# Exercício 4.5
# -------------

# Fisher Score & ReliefF

def fisher_score(X, y):
    """
    Implementação do Fisher Score por feature.
    F = sum_c n_c (mu_c - mu)^2 / sum_c n_c sigma_c^2
    Retorna vetor [n_features]
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    n, d = X.shape
    mu = np.nanmean(X, axis=0)
    num = np.zeros(d)
    den = np.zeros(d)
    classes = np.unique(y[~np.isnan(y)])
    for c in classes:
        idx = np.where(y == c)[0]
        Xc = X[idx, :]
        mu_c = np.nanmean(Xc, axis=0)
        var_c = np.nanvar(Xc, axis=0) + 1e-12
        n_c = Xc.shape[0]
        num += n_c * (mu_c - mu)**2
        den += n_c * var_c
    score = num / (den + 1e-12)
    return score

def reliefF(X, y, n_neighbors=10, n_samples=200):
    """
    ReliefF simplificado:
    - Amostra m instâncias aleatórias
    - Para cada instância:
        * encontra nearest hits (mesma classe) e misses (por classe diferente)
        * atualiza score pela diferença média
    """
    rng = np.random.RandomState(0)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(float)
    mask = ~np.isnan(y)
    X = X[mask]
    y = y[mask]
    n, d = X.shape

    m = min(n_samples, n)
    idx_s = rng.choice(n, m, replace=False)

    # Pré-ajuste vizinhos por classe
    scores = np.zeros(d)
    classes = np.unique(y)
    nbrs_all = NearestNeighbors(n_neighbors=n_neighbors+1).fit(X)

    for i in idx_s:
        xi, yi = X[i], y[i]
        # Hits
        same_mask = (y == yi)
        X_same = X[same_mask]
        if len(X_same) > 1:
            nbrs_same = NearestNeighbors(n_neighbors=min(n_neighbors+1, len(X_same))).fit(X_same)
            dists, inds = nbrs_same.kneighbors([xi], return_distance=True)
            hits = X_same[inds[0][1:]]
        else:
            hits = np.array([xi])

        # Misses: por classe diferente, ponderado pelo tamanho da classe
        miss_list = []
        for c in classes:
            if c == yi:
                continue
            Xc = X[y == c]
            if len(Xc) == 0:
                continue
            nbrs_c = NearestNeighbors(n_neighbors=min(n_neighbors, len(Xc))).fit(Xc)
            _, indc = nbrs_c.kneighbors([xi], return_distance=True)
            miss_list.append(Xc[indc[0]])
        misses = np.vstack(miss_list) if miss_list else np.array([xi])

        diff_hits = np.abs(hits - xi).mean(axis=0)
        diff_miss = np.abs(misses - xi).mean(axis=0)
        scores += (diff_miss - diff_hits) / (m + 1e-12)

    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
    return scores

# -------------
# Exercício 4.6 
# -------------

# Top features (Fisher vs ReliefF) + utilidades

def top_k_features(scores, feature_names, k=10):
    idx = np.argsort(scores)[::-1][:k]
    return [(feature_names[i], float(scores[i]), int(i)) for i in idx]

def barplot_feature_importance(pairs, title, out_path):
    names = [p[0] for p in pairs]
    vals = [p[1] for p in pairs]
    plt.figure(figsize=(10,4))
    plt.bar(range(len(names)), vals)
    plt.xticks(range(len(names)), names, rotation=45, ha="right")
    plt.ylabel("Score")
    plt.title(title)
    plt.grid(axis='y', linestyle="--", alpha=0.6)
    savefig(out_path)

# ------------------------
# Exercícios 4.4.1 e 4.6.1 
# ------------------------

# Funções de transformação de features


def transform_with_pca(X_row, pca_bundle, k):
    """
    Devolve as k componentes principais para um vetor de features (1D).
    """
    scaler, pca = pca_bundle["scaler"], pca_bundle["pca"]
    X_row = np.asarray(X_row).reshape(1, -1)
    if scaler is not None:
        X_row = scaler.transform(X_row)
    z = pca.transform(X_row)
    return z[0, :k]

def compress_with_mask(X_row, selected_indices):
    """
    Devolve as features selecionadas (e.g., top-10) para um vetor de features (1D).
    """
    X_row = np.asarray(X_row).reshape(-1)
    return X_row[selected_indices]

# ----
# Main
# ----

def main():
    pasta_base = "FORTH_TRACE_DATASET"
    participantes = list(range(0, 15))
    devices_todos = (1,2,3,4,5)

    
    todos_atividades = []
    todos_acel = []
    todos_giro = []
    todos_mag = []
    todos_devices = []
    todos_participantes = []
    todos_dados_completos = []
    fs = 50.0
    
    # -----------
    # Exercicio 2
    # -----------

    # Carregar dados e calcular módulos

    for part_id in participantes:
        try:
            dados = carregar_dados(part_id, pasta_base=pasta_base, devices=devices_todos) #Exercicio 2
        except FileNotFoundError:
            print(f"Aviso: Nenhum dado encontrado para part{part_id}")
            continue

        atividade_labels = dados[:, 11]
        device_ids = dados[:, 0]

        modulo_acel = calcular_modulo(dados[:, 1], dados[:, 2], dados[:, 3])
        modulo_giro = calcular_modulo(dados[:, 4], dados[:, 5], dados[:, 6])
        modulo_mag = calcular_modulo(dados[:, 7], dados[:, 8], dados[:, 9])

        todos_atividades.append(atividade_labels)
        todos_acel.append(modulo_acel)
        todos_giro.append(modulo_giro)
        todos_mag.append(modulo_mag)
        todos_devices.append(device_ids)
        todos_dados_completos.append(dados)
        todos_participantes.append(np.full(len(atividade_labels), part_id))

    # Concatenar todos os arrays
    todos_atividades = np.concatenate(todos_atividades)
    todos_acel = np.concatenate(todos_acel)
    todos_giro = np.concatenate(todos_giro)
    todos_mag = np.concatenate(todos_mag)
    todos_devices = np.concatenate(todos_devices)
    todos_dados_completos = np.concatenate(todos_dados_completos)

    todos_participantes = np.concatenate(todos_participantes)
    # ---------------
    # Exercício 3.1.
    # ---------------

    # Plotar boxplots (todos os sensores)
    
    boxplot_por_atividade(
        todos_acel, todos_atividades, titulo="Boxplot por atividade - Acelerómetro"
    )
    boxplot_por_atividade(
        todos_giro, todos_atividades, titulo="Boxplot por atividade - Giroscópio"
    )
    boxplot_por_atividade(
        todos_mag, todos_atividades, titulo="Boxplot por atividade - Magnetómetro"
    )
   
    # --------------
    # Exercício 3.2.
    # --------------


    # Filtrar apenas pulso direito para densidade de outliers

    mask_pulso_direito = todos_devices == 2
    todos_atividades_pulso = todos_atividades[mask_pulso_direito]
    todos_acel_pulso = todos_acel[mask_pulso_direito]
    todos_giro_pulso = todos_giro[mask_pulso_direito]
    todos_mag_pulso = todos_mag[mask_pulso_direito]
    todos_dados_pulso = todos_dados_completos[mask_pulso_direito]

    
    # Plotar boxplots (para pulso direito)
    
    boxplot_por_atividade(
        todos_acel_pulso, todos_atividades_pulso, titulo="Boxplot por atividade (pulso direito) - Acelerómetro"
    )
    boxplot_por_atividade(
        todos_giro_pulso, todos_atividades_pulso, titulo="Boxplot por atividade (pulso direito) - Giroscópio"
    )
    boxplot_por_atividade(
        todos_mag_pulso, todos_atividades_pulso, titulo="Boxplot por atividade (pulso direito) - Magnetómetro"
    )
 
    # Calcular densidade de outliers
    dens_acel = densidade_outliers_tukey(todos_acel_pulso, todos_atividades_pulso)
    dens_giro = densidade_outliers_tukey(todos_giro_pulso, todos_atividades_pulso)
    dens_mag = densidade_outliers_tukey(todos_mag_pulso, todos_atividades_pulso)

    print_densidade_legivel(dens_acel, "Acelerómetro (pulso direito)")
    print_densidade_legivel(dens_giro, "Giroscópio (pulso direito)")
    print_densidade_legivel(dens_mag, "Magnetómetro (pulso direito)")
    
    # -------------------
    # Exercício 3.3 e 3.4
    # -------------------

    # Identificação e visualização de outliers

    print("\n--- Exercício 3.3 e 3.4: Identificação e Visualização de Outliers ---")
    print("\n--- Exercício 3.3 e 3.4: Identificação e Visualização de Outliers ---")
    visualizar_outliers_multik(todos_acel_pulso, "Acelerómetro (pulso direito)", todos_atividades_pulso)
    visualizar_outliers_multik(todos_giro_pulso, "Giroscópio (pulso direito)", todos_atividades_pulso)
    visualizar_outliers_multik(todos_mag_pulso, "Magnetómetro (pulso direito)", todos_atividades_pulso)

    
    # Densidade de outliers usando Z-Score
    
    print("\n--- Densidade de Outliers usando Z-Score ---")
    for k in [3, 3.5, 4]:
        print(f"\n>>> Z-Score com k={k}")
        dens_acel_zscore = densidade_outliers_zscore(todos_acel_pulso, todos_atividades_pulso, k)
        print_densidade_legivel(dens_acel_zscore, f"Acelerómetro (pulso direito, k={k})")
        
        dens_giro_zscore = densidade_outliers_zscore(todos_giro_pulso, todos_atividades_pulso, k)
        print_densidade_legivel(dens_giro_zscore, f"Giroscópio (pulso direito, k={k})")
        
        dens_mag_zscore = densidade_outliers_zscore(todos_mag_pulso, todos_atividades_pulso, k)
        print_densidade_legivel(dens_mag_zscore, f"Magnetómetro (pulso direito, k={k})")

    # -------------
    # Exercício 3.6
    # -------------

    print("\n--- Exercício 3.6: Agrupamento com K-Means ---")
    n_clusters = 3

    # Algoritmo K-Means

    # --- Acelerômetro ---
    atribuicoes_acel, centroides_acel = kmeans(todos_acel_pulso, n_clusters)
    indices_outliers_acel, _ = identificar_outliers_kmeans(todos_acel_pulso, atribuicoes_acel, centroides_acel)
    plotar_clusters(
        todos_acel_pulso,
        atribuicoes_acel,
        centroides_acel,
        f"Acelerómetro (pulso direito) - K-Means ({n_clusters} clusters)",
        todos_atividades_pulso,
        outliers=indices_outliers_acel
    )

    # --- Giroscópio ---
    atribuicoes_giro, centroides_giro = kmeans(todos_giro_pulso, n_clusters)
    indices_outliers_giro, _ = identificar_outliers_kmeans(todos_giro_pulso, atribuicoes_giro, centroides_giro)
    plotar_clusters(
        todos_giro_pulso,
        atribuicoes_giro,
        centroides_giro,
        f"Giroscópio (pulso direito) - K-Means ({n_clusters} clusters)",
        todos_atividades_pulso,
        outliers=indices_outliers_giro
    )

    # --- Magnetómetro ---
    atribuicoes_mag, centroides_mag = kmeans(todos_mag_pulso, n_clusters)
    indices_outliers_mag, _ = identificar_outliers_kmeans(todos_mag_pulso, atribuicoes_mag, centroides_mag)
    plotar_clusters(
        todos_mag_pulso,
        atribuicoes_mag,
        centroides_mag,
        f"Magnetómetro (pulso direito) - K-Means ({n_clusters} clusters)",
        todos_atividades_pulso,
        outliers=indices_outliers_mag
    )
    # -------------
    # Exercício 3.7
    # -------------

    # Deteção de outliers com K-Means e visualização 3D
    
    print("\n--- Exercício 3.7: Outliers via K-Means (visualização 3D) ---")
    for n_clusters in [2, 3, 4]:
        print(f"\nNúmero de clusters: {n_clusters}")

        dados_acel = np.column_stack((todos_dados_pulso[:, 1], todos_dados_pulso[:, 2], todos_dados_pulso[:, 3]))
        atribuicoes_acel, centroides_acel = kmeans_multivariado(dados_acel, n_clusters)
        indices_outliers_acel, _ = identificar_outliers_kmeans(dados_acel, atribuicoes_acel, centroides_acel)
        plotar_clusters_3d(dados_acel, atribuicoes_acel, centroides_acel, indices_outliers_acel, f"Acelerómetro - {n_clusters} clusters")

        dados_giro = np.column_stack((todos_dados_pulso[:, 4], todos_dados_pulso[:, 5], todos_dados_pulso[:, 6]))
        atribuicoes_giro, centroides_giro = kmeans_multivariado(dados_giro, n_clusters)
        indices_outliers_giro, _ = identificar_outliers_kmeans(dados_giro, atribuicoes_giro, centroides_giro)
        plotar_clusters_3d(dados_giro, atribuicoes_giro, centroides_giro, indices_outliers_giro, f"Giroscópio - {n_clusters} clusters")

        dados_mag = np.column_stack((todos_dados_pulso[:, 7], todos_dados_pulso[:, 8], todos_dados_pulso[:, 9]))
        atribuicoes_mag, centroides_mag = kmeans_multivariado(dados_mag, n_clusters)
        indices_outliers_mag, _ = identificar_outliers_kmeans(dados_mag, atribuicoes_mag, centroides_mag)
        plotar_clusters_3d(dados_mag, atribuicoes_mag, centroides_mag, indices_outliers_mag, f"Magnetómetro - {n_clusters} clusters")
    
    # Densidade de outliers usando K-Means

    print("\n--- Densidade de Outliers usando K-Means ---")
    for n_clusters in [2, 3, 4]:
        print(f"\n>>> K-Means com {n_clusters} clusters")
        
        # Acelerômetro
        dados_acel_3d = np.column_stack((todos_dados_pulso[:, 1], todos_dados_pulso[:, 2], todos_dados_pulso[:, 3]))
        dens_acel_kmeans = densidade_outliers_kmeans(dados_acel_3d, todos_atividades_pulso, n_clusters)
        print_densidade_legivel(dens_acel_kmeans, f"Acelerómetro (pulso direito, {n_clusters} clusters)")
        
        # Giroscópio
        dados_giro_3d = np.column_stack((todos_dados_pulso[:, 4], todos_dados_pulso[:, 5], todos_dados_pulso[:, 6]))
        dens_giro_kmeans = densidade_outliers_kmeans(dados_giro_3d, todos_atividades_pulso, n_clusters)
        print_densidade_legivel(dens_giro_kmeans, f"Giroscópio (pulso direito, {n_clusters} clusters)")
        
        # Magnetómetro
        dados_mag_3d = np.column_stack((todos_dados_pulso[:, 7], todos_dados_pulso[:, 8], todos_dados_pulso[:, 9]))
        dens_mag_kmeans = densidade_outliers_kmeans(dados_mag_3d, todos_atividades_pulso, n_clusters)
        print_densidade_legivel(dens_mag_kmeans, f"Magnetómetro (pulso direito, {n_clusters} clusters)")

    # -----------
    # Exercicio 4
    # -----------
    
    print("\\n--- Objetivo 4: Extração de features, PCA e Feature Selection (guardado em outputs/) ---")
    
    # -------------
    # Exercício 4.1
    # -------------

    # Normalidade + comparação de médias por atividade

    results_41 = {}
    for name, vec in [("acc", todos_acel), ("gyr", todos_giro), ("mag", todos_mag)]:
        ks_res = ks_normality_by_activity(vec, todos_atividades)
        test_res = test_means_across_activities(vec, todos_atividades)
        results_41[name] = {"ks": ks_res, "means_test": test_res}

    # Guardar 4.1 em CSV
    rows = []
    for name in results_41:
        for a, (stat, p, n) in results_41[name]["ks"].items():
            rows.append([name, a, stat, p, n])
    save_csv(os.path.join(outdir, "4_1_ks_normality.csv"), ["variable","activity","ks_stat","pvalue","n"], rows)

    rows = []
    for name in results_41:
        rows.append([name, results_41[name]["means_test"]["method"], results_41[name]["means_test"]["pvalue"]])
    save_csv(os.path.join(outdir, "4_1_means_comparison.csv"), ["variable","method","pvalue"], rows)
    
    # -------------
    # Exercício 4.2
    # -------------

    # Extração de features por janelas nos módulos

    print("A extrair features com janelas...")
    
    # Passamos 'todos_participantes' para a função
    # Recebemos 'p_win' (participante por janela) de volta
    X_acc, y_win, p_win, feat_names = sliding_window_features(todos_acel, todos_atividades, todos_participantes, todos_devices, fs)
    X_gyr, _, _, _ = sliding_window_features(todos_giro, todos_atividades, todos_participantes, todos_devices, fs)
    X_mag, _, _, _ = sliding_window_features(todos_mag, todos_atividades, todos_participantes, todos_devices, fs)

    # Concatenar features (se existirem dados)
    if X_acc.size and X_gyr.size and X_mag.size:
        X_all = np.hstack([X_acc, X_gyr, X_mag])
        feat_names_all = [f"acc_{n}" for n in feat_names] + \
                         [f"gyr_{n}" for n in feat_names] + \
                         [f"mag_{n}" for n in feat_names]
        
        # Preparar linhas para o CSV incluindo o PARTICIPANTE
        # Header: [participante, label, acc_mean, ..., gyr_mean, ..., mag_mean...]
        header = ["participante", "label"] + feat_names_all
        
        rows = []
        for i in range(len(y_win)):
            # Construir a linha: [PartID, Label, Features...]
            row = [int(p_win[i]), int(y_win[i])] + list(X_all[i])
            rows.append(row)

        save_csv(os.path.join(outdir, "4_2_features_windows.csv"), header, rows)
        print("Features extraídas e guardadas em 4_2_features_windows.csv")
    else:
        print("Erro: Não foi possível extrair features (arrays vazios).")

    # --------------------
    # Exercícios 4.3 & 4.4
    # --------------------

    # PCA, variância explicada e nº de componentes
    if X_all.shape[0] >= 2:
        pca_bundle = run_pca(X_all, var_norm=True)
        pca = pca_bundle["pca"]
        Z = pca_bundle["Z"]
        k75, cum = num_components_for_variance(pca, threshold=0.75)

        
        plot_variance_explained(pca, os.path.join(outdir, "4_4_pca_variance.png"))

       
        save_csv(os.path.join(outdir, "4_3_pca_components.csv"), None, Z.tolist())
        save_csv(os.path.join(outdir, "4_4_pca_k75.csv"), ["k_for_75pct"], [[k75]])

        # 4.4.1 
        # Exemplo de features comprimidas via PCA num instante =====
        if Z.shape[0] > 0:
            z_example = Z[0, :k75].tolist()
            save_csv(os.path.join(outdir, "4_4_1_pca_example_row.csv"), [f"pc{i+1}" for i in range(k75)], [z_example])

        # -------------
        # Exercício 4.5
        # -------------
        
        # Fisher Score & ReliefF

        fisher = fisher_score(X_all, y_win)
        relief = reliefF(X_all, y_win, n_neighbors=10, n_samples=min(500, len(y_win)))

        save_csv(os.path.join(outdir, "4_5_scores_fisher.csv"), feat_names_all, [fisher.tolist()])
        save_csv(os.path.join(outdir, "4_5_scores_reliefF.csv"), feat_names_all, [relief.tolist()])

        # -------------
        # Exercício 4.6
        # -------------

        # Top-10 features e comparação

        top10_fisher = top_k_features(fisher, feat_names_all, k=10)
        top10_relief = top_k_features(relief, feat_names_all, k=10)

        
        save_csv(os.path.join(outdir, "4_6_top10_fisher.csv"), ["feature","score","index"], top10_fisher)
        save_csv(os.path.join(outdir, "4_6_top10_reliefF.csv"), ["feature","score","index"], top10_relief)

       
        barplot_feature_importance(top10_fisher, "Top-10 Fisher Score", os.path.join(outdir, "4_6_top10_fisher.png"))
        barplot_feature_importance(top10_relief, "Top-10 ReliefF", os.path.join(outdir, "4_6_top10_reliefF.png"))

        # -------------
        # Exercício 4.6
        # -------------

        # 4.6.1: Exemplo de compressão por seleção (top-10 de Fisher)

        selected_idx = [idx for (_, _, idx) in top10_fisher]
        if X_all.shape[0] > 0:
            sel_example = compress_with_mask(X_all[0], selected_idx).tolist()
            save_csv(os.path.join(outdir, "4_6_1_selection_example_row.csv"),
                     [feat_names_all[i] for i in selected_idx], [sel_example])



# ------------------------------
# Executar main
# ------------------------------

if __name__ == "__main__":
    main()