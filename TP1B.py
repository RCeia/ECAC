import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 0. FUNÇÃO DE CARREGAMENTO DE DADOS
# =============================================================================
def carregar_dados_filtrados(part_id, pasta_base="FORTH_TRACE_DATASET", devices=(1,2,3,4,5)):
    """
    Carrega os ficheiros de um participante específico.
    Lê linha a linha e apenas guarda se a atividade (última coluna) for <= 7.
    """
    rows = []
    
    for dev in devices:
        # Constrói o caminho: ex: FORTH_TRACE_DATASET/part1/part1dev1.csv
        path = os.path.join(pasta_base, f"part{part_id}", f"part{part_id}dev{dev}.csv")
        
        if not os.path.isfile(path):
            continue
            
        try:
            with open(path, "r") as f:
                reader = csv.reader(f)
                for line in reader:
                    if not line:
                        continue
                    try:
                        # Converter valores para float
                        vals = [float(x) for x in line]
                        
                        # A atividade é a última coluna (-1)
                        atividade = vals[-1]
                        
                        # --- FILTRO: APENAS ATIVIDADES 1 A 7 ---
                        if atividade <= 7:
                            rows.append(vals)
                            
                    except ValueError:
                        continue # Ignora linhas com erros de formatação
        except Exception as e:
            print(f"Erro ao ler ficheiro {path}: {e}")

    if len(rows) == 0:
        return None
        
    return np.array(rows)

# =============================================================================
# 1.1 FUNÇÃO DE PLOT (GRÁFICO DE BARRAS)
# =============================================================================
def plot_distribuicao_atividades(dataset, output_dir="outputsB"):
    """
    Recebe o dataset completo (numpy array), conta as ocorrências
    da última coluna (atividade) e gera um gráfico de barras.
    """
    if dataset is None or len(dataset) == 0:
        print("Dataset vazio. Não é possível gerar o gráfico.")
        return

    # Extrair a coluna das atividades (última coluna)
    coluna_atividades = dataset[:, -1]
    
    # Contar quantas amostras existem por atividade
    atividades_unicas, contagens = np.unique(coluna_atividades, return_counts=True)
    
    # --- Criação do Gráfico ---
    plt.figure(figsize=(10, 6))
    
    # Barras
    barras = plt.bar(atividades_unicas, contagens, color='steelblue', edgecolor='black', alpha=0.8)
    
    # Estética
    plt.title('Total de Amostras por Atividade (1 a 7)', fontsize=14, fontweight='bold')
    plt.xlabel('ID da Atividade', fontsize=12)
    plt.ylabel('Número de Amostras', fontsize=12)
    plt.xticks(atividades_unicas) # Força a mostrar todos os IDs no eixo X
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Adicionar o valor exato no topo de cada barra
    for barra in barras:
        height = barra.get_height()
        plt.text(barra.get_x() + barra.get_width()/2., height,
                 f'{int(height)}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Guardar
    os.makedirs(output_dir, exist_ok=True)
    caminho_figura = os.path.join(output_dir, "comparacao_atividades.png")
    
    plt.tight_layout()
    plt.savefig(caminho_figura, dpi=150)
    plt.show()
    
    print(f"Gráfico guardado com sucesso em: {caminho_figura}")
    
    # Imprimir resumo no terminal
    print("\n--- Resumo das Contagens ---")
    for ativ, count in zip(atividades_unicas, contagens):
        print(f"Atividade {int(ativ)}: {count} amostras")

# =============================================================================
# MAIN
# =============================================================================
def main():
    # Exercício 0
    NUM_PARTICIPANTES = 15 # Ajuste conforme o seu dataset real (ex: 10, 18, etc.)
    PASTA_DADOS = "FORTH_TRACE_DATASET"
    
    print(f"A iniciar carregamento de dados da pasta: {PASTA_DADOS}")
    print("Filtrando apenas atividades <= 7...")
    
    dados_todos = []

    # Loop pelos participantes
    for p_id in range(1, NUM_PARTICIPANTES + 1):
        dados_part = carregar_dados_filtrados(p_id, pasta_base=PASTA_DADOS)
        
        if dados_part is not None:
            dados_todos.append(dados_part)
            print(f"-> Participante {p_id}: OK ({len(dados_part)} linhas)")
        else:
            print(f"-> Participante {p_id}: Sem dados.")
    
    # Exercício 1.1
    
    # Verificar e Concatenar
    if len(dados_todos) > 0:
        dataset_final = np.concatenate(dados_todos, axis=0)
        print(f"\nCarregamento concluído. Total de linhas acumuladas: {len(dataset_final)}")
        
        # Chama a função de plot
        plot_distribuicao_atividades(dataset_final)
    else:
        print("\nERRO: Não foi possível carregar nenhuns dados. Verifique o caminho da pasta.")

if __name__ == "__main__":
    main()