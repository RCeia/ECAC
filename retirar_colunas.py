import csv
import os

def limpar_dados_para_deploy(ficheiro_alvo):
    """
    Lê o ficheiro CSV, extrai apenas as colunas de sensores (índices 1 a 9)
    e sobrescreve o ficheiro original apenas com esses dados numéricos.
    
    Input esperado: [DeviceID, Ax, Ay, Az, Gx, Gy, Gz, Mx, My, Mz, Timestamp, (Label)]
    Output: [Ax, Ay, Az, Gx, Gy, Gz, Mx, My, Mz]
    """
    if not os.path.exists(ficheiro_alvo):
        print(f"ERRO: O ficheiro '{ficheiro_alvo}' não existe.")
        return

    print(f"A ler e limpar '{ficheiro_alvo}'...")
    
    linhas_limpas = []
    
    try:
        # 1. Ler o conteúdo original
        with open(ficheiro_alvo, 'r') as f_in:
            reader = csv.reader(f_in)
            
            for i, row in enumerate(reader):
                if not row: continue # Ignorar linhas vazias
                
                try:
                    # Validar se a linha tem tamanho suficiente
                    if len(row) < 10:
                        print(f"   Aviso (Linha {i+1}): Linha demasiado curta ({len(row)} colunas). Ignorada.")
                        continue
                    
                    # --- O PASSO CRÍTICO ---
                    # Ignora a coluna 0 (Device ID)
                    # Pega nas colunas 1 a 9 (Acc, Gyr, Mag)
                    # Ignora da coluna 10 em diante (Timestamp, Label, etc.)
                    dados_sensores = row[1:10]
                    
                    # Validar se são números (opcional, mas recomendado para não quebrar o deploy)
                    # Isto garante que não escrevemos lixo no ficheiro final
                    _ = [float(x) for x in dados_sensores] 
                    
                    linhas_limpas.append(dados_sensores)
                    
                except ValueError:
                    print(f"   Aviso (Linha {i+1}): Contém valores não numéricos. Ignorada.")
                    continue

        if len(linhas_limpas) == 0:
            print("Erro: Não foram encontradas linhas válidas.")
            return

        # 2. Sobrescrever o ficheiro original com os dados limpos
        with open(ficheiro_alvo, 'w', newline='') as f_out:
            writer = csv.writer(f_out)
            writer.writerows(linhas_limpas)
            
        print(f"\n✅ Sucesso! O ficheiro '{ficheiro_alvo}' foi atualizado.")
        print(f"   Linhas originais processadas: {i+1}")
        print(f"   Linhas limpas guardadas: {len(linhas_limpas)}")
        print("   Formato atual: 9 colunas numéricas (pronto para o Exercício 6).")

    except Exception as e:
        print(f"Ocorreu um erro crítico: {e}")

if __name__ == "__main__":
    # Nome do ficheiro a limpar
    TARGET_FILE = "teste.csv"
    
    # Se o ficheiro não existir, cria-o com os dados que enviou para teste
    if not os.path.exists(TARGET_FILE):
        print(f"Criando '{TARGET_FILE}' com os dados de exemplo fornecidos...")
        dados_exemplo = """5,-0.45242,9.6193,-1.476,0.081159,-0.51129,0.97707,-0.69091,0.58283,-0.21739,1499.2
5,-0.46449,9.6193,-1.4636,0.26139,-0.28353,0.54912,-0.70707,0.55888,-0.21304,1518.7
5,-0.44047,9.6313,-1.4887,0.39979,-0.43544,0.56921,-0.68283,0.55888,-0.21957,1538.2
5,-0.50155,9.6306,-1.4997,0.20226,-0.45372,0.47345,-0.68485,0.58283,-0.21739,1557.8,1
5,-0.51362,9.6306,-1.4872,0.20024,-0.25194,0.69928,-0.69697,0.57086,-0.21304,1577.3,1"""
        with open(TARGET_FILE, "w") as f:
            f.write(dados_exemplo)

    # Executar limpeza
    limpar_dados_para_deploy(TARGET_FILE)