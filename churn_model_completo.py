import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
import warnings
import gc

warnings.filterwarnings('ignore')

# Funções Auxiliares para Monitoramento e Configuração

def check_memory_usage(df_name: str, dataframe: pd.DataFrame):
    """
    Imprime o uso de memória de um DataFrame em MB.
    """
    if not isinstance(dataframe, pd.DataFrame):
        print(f"'{df_name}' não é um DataFrame válido para verificar o uso de memória.")
        return
    mem_usage = dataframe.memory_usage(deep=True).sum() / (1024**2) # em MB
    print(f"Uso de memória para '{df_name}': {mem_usage:.2f} MB")

def setup_plot_style():
    """Configura o estilo dos plots para melhor visualização."""
    plt.style.use('seaborn-v0_8-darkgrid')
    
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.facecolor'] = '#333333' # Fundo cinza escuro
    plt.rcParams['figure.facecolor'] = '#333333' # Fundo da figura cinza escuro
    plt.rcParams['text.color'] = 'white' # Texto branco para contraste
    plt.rcParams['axes.labelcolor'] = 'white' # Rótulos dos eixos em branco
    plt.rcParams['xtick.color'] = 'white' # Ticks em branco
    plt.rcParams['ytick.color'] = 'white' # Ticks em branco
    plt.rcParams['grid.color'] = '#666666' # Linhas de grade em cinza médio
    plt.rcParams['figure.dpi'] = 100

setup_plot_style()

# 1.Carregamento de Dados

def load_data(file_path: str, sample_fraction: float = 1.0) -> pd.DataFrame:
    """
    Carrega o dataset a partir de um arquivo Excel.
    Lida com erros de arquivo não encontrado e outros erros de leitura.
    Permite carregar apenas uma fração das linhas para teste.
    """
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        if sample_fraction < 1.0:
            original_rows = df.shape[0]
            df = df.sample(frac=sample_fraction, random_state=42).reset_index(drop=True)
            print(f"--- Dataset reduzido para {df.shape[0]} linhas ({sample_fraction*100:.1f}% do original de {original_rows} linhas) para fins de teste. ---")
        
        print(f"--- SUCESSO: Dados carregados de '{file_path}' ---")
        check_memory_usage("DataFrame Original", df)
        
        print("\nDEBUG: Colunas do DataFrame recém-carregado:")
        print(df.columns.tolist())
        
        return df
    except FileNotFoundError:
        print(f"ERRO: O arquivo '{file_path}' não foi encontrado. Por favor, verifique o caminho.")
        print("Certifique-se de que o arquivo está no ambiente do Colab (upload manual ou Google Drive montado).")
        return pd.DataFrame()
    except Exception as e:
        print(f"ERRO ao carregar o arquivo: {e}")
        return pd.DataFrame()

# Defina o caminho do arquivo
file_name = 'WA_Fn-UseC_-Telco-Customer-Churn.xlsx'

df = load_data(file_name, sample_fraction=1.0)

if df.empty:
    print("Não foi possível carregar os dados. Encerrando a execução.")
    exit()

# 2.Data Wrangling e Pré-processamento

def preprocess_data(data: pd.DataFrame, expected_churn_col: str = 'Churn') -> pd.DataFrame:
    """
    Realiza as etapas de limpeza e pré-processamento dos dados,
    com otimização de memória e tratamento de tipos de dados.
    Inclui verificação robusta para a coluna de churn.
    """
    print("\n--- Iniciando Pré-processamento de Dados ---")
    df_processed = data.copy()

    actual_churn_col_name = None
    if expected_churn_col in df_processed.columns:
        actual_churn_col_name = expected_churn_col
        print(f"  > Coluna de churn '{expected_churn_col}' encontrada. Proseguindo.")
    else:
        churn_candidates = [col for col in df_processed.columns if 'churn' in str(col).lower()]
        if churn_candidates:
            actual_churn_col_name = churn_candidates[0]
            print(f"  > Coluna de churn '{expected_churn_col}' não encontrada. Usando '{actual_churn_col_name}' como alternativa.")
        else:
            print("  ERRO GRAVE: Nenhuma coluna de churn foi encontrada no DataFrame.")
            print(f"  Colunas disponíveis: {df_processed.columns.tolist()}")
            return pd.DataFrame()
    
    if actual_churn_col_name != 'churn_status':
        df_processed.rename(columns={actual_churn_col_name: 'churn_status'}, inplace=True)
        print(f"  > Coluna '{actual_churn_col_name}' renomeada para 'churn_status'.")

    target_col = 'churn_status'

    # 2.1 Tratamento de 'TotalCharges' e NaNs
    df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
    initial_rows = df_processed.shape[0]
    df_processed.dropna(inplace=True)
    rows_dropped = initial_rows - df_processed.shape[0]
    if rows_dropped > 0:
        print(f"  > Removidas {rows_dropped} linhas com valores ausentes (NaNs) após conversão de 'TotalCharges'.")
    check_memory_usage("df_processed após dropna", df_processed)

    # 2.2 Otimização de tipos de dados para colunas numéricas
    for col in df_processed.select_dtypes(include=['int64', 'float64']).columns:
        if col == 'customer_id':
            continue
        if df_processed[col].dtype == 'int64':
            df_processed[col] = pd.to_numeric(df_processed[col], downcast='integer')
        elif df_processed[col].dtype == 'float64':
            df_processed[col] = pd.to_numeric(df_processed[col], downcast='float')
    print("  > Tipos de dados numéricos existentes otimizados (downcast).")
    check_memory_usage("df_processed após downcast numérico", df_processed)

    # 2.3 Renomear outras colunas para snake_case
    column_mapping = {
        'customerID': 'customer_id', 'gender': 'gender', 'SeniorCitizen': 'senior_citizen',
        'Partner': 'has_partner', 'Dependents': 'has_dependents', 'tenure': 'months_with_service',
        'PhoneService': 'phone_service', 'MultipleLines': 'multiple_lines', 'InternetService': 'internet_service',
        'OnlineSecurity': 'online_security', 'OnlineBackup': 'online_backup', 'DeviceProtection': 'device_protection',
        'TechSupport': 'tech_support', 'StreamingTV': 'streaming_tv', 'StreamingMovies': 'streaming_movies',
        'Contract': 'contract_type', 'PaperlessBilling': 'paperless_billing', 'PaymentMethod': 'payment_method',
        'MonthlyCharges': 'monthly_charges', 'TotalCharges': 'total_charges'
    }
    
    cols_to_rename = {k: v for k, v in column_mapping.items() if k in df_processed.columns and k != target_col}
    
    if cols_to_rename:
        df_processed.rename(columns=cols_to_rename, inplace=True)
        print("  > Outras colunas renomeadas para snake_case.")
    else:
        print("  > Nenhuma outra coluna para renomear encontrada no mapping.")

    # 2.4 TRATAMENTO FINAL DA COLUNA ALVO churn_status
    if df_processed[target_col].dtype == 'object':
        df_processed[target_col] = df_processed[target_col].map({'Yes': 1, 'No': 0})
        print("  > Coluna 'churn_status' mapeada de 'Yes'/'No' para 1/0 (segunda verificação).")
    
    if df_processed[target_col].dtype != 'int':
        if df_processed[target_col].isnull().any():
            print("  AVISO: 'churn_status' contém NaNs após mapeamento. Removendo linhas com NaNs na coluna alvo.")
            df_processed.dropna(subset=[target_col], inplace=True)
        df_processed[target_col] = df_processed[target_col].astype(int)
        print("  > Coluna 'churn_status' garantida como tipo inteiro (0/1) (segunda verificação).")

    # 2.5 Transformar outras variáveis categóricas em dummies
    categorical_cols = [
        col for col in df_processed.columns
        if df_processed[col].dtype == 'object' and col not in ['customer_id', target_col]
    ]
    
    if not categorical_cols:
        print("  > Nenhuma coluna categórica para One-Hot Encoding encontrada (além de customer_id/churn_status).")
    else:
        print(f"  > Colunas categóricas para One-Hot Encoding: {categorical_cols}")
        print("  > Verificando cardinalidade das colunas categóricas:")
        for col in categorical_cols:
            n_unique = df_processed[col].nunique()
            print(f"    - '{col}': {n_unique} valores únicos.")
            if n_unique > 50:
                print(f"      AVISO: Coluna '{col}' tem alta cardinalidade ({n_unique}). "
                      "Isso pode gerar muitas colunas dummy e consumir muita memória.")

        for col in categorical_cols:
            df_processed[col] = df_processed[col].astype('category')
        print("  > Colunas categóricas convertidas para o tipo 'category' para otimização.")
        check_memory_usage("df_processed antes de get_dummies", df_processed)

        df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True, sparse=True, dtype=int)
        print(f"  > Variáveis categóricas transformadas em variáveis dummy (esparsas=True).")
        check_memory_usage("df_processed após get_dummies (esparsas)", df_processed)

    print("--- Pré-processamento de Dados Concluído ---")
    return df_processed

df_preprocessed = preprocess_data(df, expected_churn_col='Churn')

del df
gc.collect()
print("\n--- DataFrame original 'df' removido da memória e GC.collect() executado. ---")
check_memory_usage("df_preprocessed (final)", df_preprocessed)


target_col = 'churn_status'
if df_preprocessed.empty or target_col not in df_preprocessed.columns:
    print(f"ERRO: A coluna alvo '{target_col}' não foi encontrada no DataFrame pré-processado. Encerrando.")
    exit()
elif df_preprocessed[target_col].dtype not in [np.int64, np.int32, np.int16, np.int8]:
     print(f"ERRO: A coluna alvo '{target_col}' não é do tipo inteiro. Tipo atual: {df_preprocessed[target_col].dtype}. Encerrando.")
     exit()

# 3.Análise Exploratória de Dados (EDA) e Visualização 

def plot_top_correlations(data: pd.DataFrame, target: str, num_features: int = 10):
    """
    Calcula e plota as top N correlações absolutas com a variável alvo.
    Retorna a lista das features mais correlacionadas.
    """
    print(f"\n--- Iniciando Análise de Correlação com '{target}' ---")
    if target not in data.columns:
        print(f"ERRO: A coluna alvo '{target}' não existe no DataFrame para cálculo de correlação.")
        return []

    data_for_numeric_ops = data.copy() 
    
    for col in data_for_numeric_ops.columns:
        if isinstance(data_for_numeric_ops[col].dtype, pd.SparseDtype):
            data_for_numeric_ops[col] = data_for_numeric_ops[col].sparse.to_dense()
            print(f"  > Coluna '{col}' convertida de esparsa para densa.")

    numeric_df_for_corr = data_for_numeric_ops.select_dtypes(include=np.number)
    
    cols_to_exclude_from_corr = ['customer_id']
    numeric_df_for_corr = numeric_df_for_corr.drop(columns=[col for col in cols_to_exclude_from_corr if col in numeric_df_for_corr.columns], errors='ignore')

    if target not in numeric_df_for_corr.columns:
        print(f"AVISO: A coluna alvo '{target}' não é numérica ou foi removida para cálculo de correlação.")
        return []

    correlations = numeric_df_for_corr.corr(method='spearman')[target].abs().sort_values(ascending=False)
    
    top_correlations = correlations[correlations.index != target].head(num_features)

    if top_correlations.empty:
        print(f"AVISO: Não foram encontradas correlações significativas com '{target}'.")
        return []

    plt.figure(figsize=(12, 7))
    sns.barplot(x=top_correlations.index, y=top_correlations.values, palette=['#FF0000', '#CC0000', '#990000', '#660000', '#330000']) 
    plt.title(f'Top {num_features} Correlações Absolutas com Churn', fontsize=18, color='red')
    plt.xlabel('Variáveis', fontsize=14, color='white')
    plt.ylabel('Coeficiente de Correlação (Spearman)', fontsize=14, color='white')
    plt.xticks(rotation=45, ha='right', fontsize=12, color='white')
    plt.yticks(fontsize=12, color='white')
    plt.grid(axis='y', linestyle='--', alpha=0.7, color='gray')
    plt.tight_layout()
    plt.show()
    print("--- Análise de Correlação Concluída ---")
    return top_correlations.index.tolist()

selected_features_for_model = plot_top_correlations(df_preprocessed, target_col, num_features=10)

def plot_churn_distribution(data: pd.DataFrame, target: str):
    print(f"\n--- Iniciando Plotagem de Distribuição de '{target}' ---")
    if target not in data.columns:
        print(f"ERRO: A coluna alvo '{target}' não existe no DataFrame para plotar a distribuição.")
        return

    churn_counts = data[target].value_counts()
    churn_labels = {0: 'Não Churn', 1: 'Churn'}
    labels_order = [churn_labels[0], churn_labels[1]]
    counts_order = [churn_counts.get(0, 0), churn_counts.get(1, 0)]

    plt.figure(figsize=(10, 6))
    bars = sns.barplot(x=labels_order, y=counts_order, palette=['#D3D3D3', '#FF0000']) 
    plt.title(f'Distribuição de Clientes por Churn', fontsize=18, color='red')
    plt.ylabel('Número de Clientes', fontsize=14, color='white')
    plt.xlabel('Status de Churn', fontsize=14, color='white')
    plt.xticks(fontsize=12, color='white')
    plt.yticks(fontsize=12, color='white')
    
    for i, bar in enumerate(bars.patches):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                 f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=12, 
                 color='black' if i == 0 else 'red')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7, color='gray')
    plt.tight_layout()
    plt.show()
    print("--- Plotagem de Distribuição Concluída ---")

plot_churn_distribution(df_preprocessed, target_col)

# 4.PCA Análise de Componentes Principais

def perform_pca_and_plot(data: pd.DataFrame, target: str, features_list: list):
    print("\n--- Iniciando PCA para Visualização ---")
    if not features_list:
        print("AVISO: Nenhuma feature selecionada para PCA. Pulando PCA.")
        return
    
    X_pca_data = data[features_list].copy()
    y_pca_data = data[target]

    for col in X_pca_data.columns:
        if isinstance(X_pca_data[col].dtype, pd.SparseDtype):
            X_pca_data[col] = X_pca_data[col].sparse.to_dense()
            print(f"  > Coluna '{col}' para PCA convertida de esparsa para densa.")

    scaler = StandardScaler()
    try:
        X_scaled = scaler.fit_transform(X_pca_data)
        print("  > Dados para PCA escalados.")
    except Exception as e:
        print(f"ERRO ao escalar dados para PCA: {e}. Verifique se as features são numéricas e não têm NaNs.")
        return

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print(f"  > PCA concluído. Componentes explicam: {pca.explained_variance_ratio_.sum()*100:.1f}% da variância.")

    colors_pca = {0: '#D3D3D3', 1: '#FF0000'}
    
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_pca_data, 
                    palette=[colors_pca[0], colors_pca[1]], 
                    alpha=0.7, s=80, legend='full')
    
    plt.xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12, color='white')
    plt.ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12, color='white')
    plt.title(f'PCA - Visualização de Clientes', fontsize=18, color='red')
    plt.xticks(fontsize=10, color='white')
    plt.yticks(fontsize=10, color='white')
    plt.grid(True, linestyle='--', alpha=0.6, color='gray')
    
    handles, labels = plt.gca().get_legend_handles_labels()
    legend_labels_map = {'0': 'Não Churn', '1': 'Churn'}
    custom_labels = [legend_labels_map[label] for label in labels]
    
    plt.legend(handles, custom_labels, title='Status de Churn', loc='best', fontsize=10, 
               title_fontsize=12, facecolor='#333333', edgecolor='white', labelcolor='white')
    
    plt.tight_layout()
    plt.show()
    print("--- PCA para Visualização Concluído ---")

perform_pca_and_plot(df_preprocessed, target_col, selected_features_for_model)

# 5. Modelagem com Regressão Logística

def train_and_evaluate_model(data: pd.DataFrame, target: str, features: list):
    print("\n--- Iniciando Modelagem e Avaliação com Regressão Logística ---")
    if not features:
        print("ERRO: Nenhuma feature selecionada para o modelo. Não é possível treinar.")
        return None, None, None

    X = data[features].copy()
    y = data[target]

    for col in X.columns:
        if isinstance(X[col].dtype, pd.SparseDtype):
            X[col] = X[col].sparse.to_dense()
            print(f"  > Coluna '{col}' para modelagem convertida de esparsa para densa.")

    scaler = StandardScaler()
    try:
        X_scaled = scaler.fit_transform(X)
        print("  > Features escaladas (StandardScaler).")
    except Exception as e:
        print(f"ERRO ao escalar features para o modelo: {e}. Verifique a integridade dos dados.")
        return None, None, None

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    print(f"  > Dados divididos: Treino ({X_train.shape[0]} amostras), Teste ({X_test.shape[0]} amostras)." if X_train.size > 0 else "  > Dados vazios ou insuficientes para divisão.")

    model = LogisticRegression(max_iter=2000, random_state=42, solver='liblinear', class_weight='balanced')
    model.fit(X_train, y_train)
    print("  > Modelo de Regressão Logística treinado.")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n--- Relatório de Classificação ---")
    print(classification_report(y_test, y_pred))

    auc_roc = roc_auc_score(y_test, y_proba)
    print(f'AUC-ROC: {auc_roc:.4f}')

    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    plt.figure(figsize=(9, 7))
    plt.plot(fpr, tpr, color='red', lw=2, label=f'Curva ROC (AUC = {auc_roc:.2f})')
    plt.plot([0, 1], [0, 1], color='white', lw=2, linestyle='--', label='Classificação Aleatória')
    plt.xlabel('Taxa de Falsos Positivos (FPR)', fontsize=12, color='white')
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)', fontsize=12, color='white')
    plt.title('Curva ROC', fontsize=18, color='red')
    plt.xticks(fontsize=10, color='white')
    plt.yticks(fontsize=10, color='white')
    plt.grid(True, linestyle='--', alpha=0.6, color='gray')
    plt.legend(loc='lower right', fontsize=10, facecolor='#333333', edgecolor='white', labelcolor='white')
    plt.tight_layout()
    plt.show()
    print("--- Modelagem e Avaliação Concluídas ---")

    return y_test, y_pred, y_proba

y_test_model, y_pred_model, y_proba_model = train_and_evaluate_model(df_preprocessed, target_col, selected_features_for_model)

# 6. Gráfico de Desempenho 

def plot_performance_metrics(y_true, y_pred):
    print("\n--- Iniciando Plotagem de Métricas de Desempenho ---")
    if y_true is None or y_pred is None:
        print("AVISO: Dados de teste ou previsão não disponíveis para plotar métricas de desempenho.")
        return

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    metrics = {'Precisão': precision, 'Revocação': recall, 'F1-Score': f1}
    values = list(metrics.values())
    
    colors_pie = ['#FF0000', '#FFFFFF', '#666666'] #

    fig, ax = plt.subplots(figsize=(9, 7))
    
    wedges, texts, autotexts = ax.pie(values, autopct='%1.1f%%', startangle=90, colors=colors_pie,
                                       pctdistance=0.85, textprops={'fontsize': 12})
    
    for i, autotext in enumerate(autotexts):
       
        autotext.set_color('white' if i == 0 else '#333333' if i == 1 else 'white') 

    centre_circle = plt.Circle((0,0),0.60,fc='#333333') 
    fig.gca().add_artist(centre_circle)
    
    ax.set_title('Métricas de Desempenho do Modelo', fontsize=18, color='red')
    ax.axis('equal')
    
    labels_legend = [f'{k}: {v:.2f}' for k, v in metrics.items()]
    
    ax.legend(wedges, labels_legend,
              title="Métricas",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1),
              fontsize=11,
              facecolor='#333333', 
              edgecolor='white', 
              labelcolor='white') 
              
    plt.tight_layout()
    plt.show()
    print("--- Plotagem de Métricas de Desempenho Concluída ---")

plot_performance_metrics(y_test_model, y_pred_model)

del df_preprocessed
gc.collect()
print("\n--- DataFrame pré-processado 'df_preprocessed' removido da memória. ---")
print("\n--- Fim da Execução Completa do Script de Análise de Churn ---")
