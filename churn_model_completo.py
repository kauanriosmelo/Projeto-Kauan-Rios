import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA

# Carregar os dados
df = pd.read_excel('WA_Fn-UseC_-Telco-Customer-Churn.xlsx')

# Data Wrangling
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# Renomear colunas
df.rename(columns={
    'customerID': 'Customer_ID',
    'gender': 'Gender',
    'SeniorCitizen': 'Senior_Citizen',
    'Partner': 'Has_Partner',
    'Dependents': 'Has_Dependents',
    'tenure': 'Months_With_Service',
    'PhoneService': 'Phone_Service',
    'MultipleLines': 'Multiple_Lines',
    'InternetService': 'Internet_Service',
    'OnlineSecurity': 'Online_Security',
    'OnlineBackup': 'Online_Backup',
    'DeviceProtection': 'Device_Protection',
    'TechSupport': 'Tech_Support',
    'StreamingTV': 'Streaming_TV',
    'StreamingMovies': 'Streaming_Movies',
    'Contract': 'Contract_Type',
    'PaperlessBilling': 'Paperless_Billing',
    'PaymentMethod': 'Payment_Method',
    'MonthlyCharges': 'Monthly_Charges',
    'TotalCharges': 'Total_Charges',
    'Churn': 'Churn_Status'
}, inplace=True)

# Transformar variáveis categóricas em dummies
df = pd.get_dummies(df, drop_first=True)

# Identificar a coluna alvo
target_col = [col for col in df.columns if 'Churn_Status' in col][0]

# Correlação e visualização das top variáveis
num_top_features = 10
top_correlations = df.corr()[target_col].abs().sort_values(ascending=False).iloc[1:num_top_features + 1]

plt.figure(figsize=(10, 6))
top_correlations.plot(kind='bar', color='darkred')
plt.title('Top 10 Correlações com Churn_Status', fontsize=16, color='red')
plt.xlabel('Variáveis', fontsize=12, color='red')
plt.ylabel('Coeficiente de Correlação', fontsize=12, color='red')
plt.xticks(rotation=45, fontsize=10, ha='right', color='red')
plt.yticks(fontsize=10, color='red')
plt.grid(axis='y', linestyle='--', alpha=0.7, color='gray')
plt.gca().set_facecolor('black')
plt.show()

# Gráfico de distribuição do churn
churn_counts = df[target_col].value_counts()
churn_labels = ['Não Realizou Churn', 'Realizou Churn']

plt.figure(figsize=(10, 6))
plt.bar(churn_labels, churn_counts.values, color=['lightcoral', 'darkred'])
plt.title('Distribuição do Churn', fontsize=16, color='red')
plt.ylabel('Número de Clientes', fontsize=12, color='red')
plt.xlabel('Status de Churn', fontsize=12, color='red')
plt.xticks(fontsize=10, color='red')
plt.yticks(fontsize=10, color='red')
for i, value in enumerate(churn_counts.values):
    plt.text(i, value + 50, str(value), ha='center', fontsize=12, color='red')
plt.grid(axis='y', linestyle='--', alpha=0.7, color='gray')
plt.gca().set_facecolor('black')
plt.show()

# Seleção de variáveis e separação
selected_features = top_correlations.index.tolist()
X = df[selected_features]
y = df[target_col]

# PCA - Análise de Componentes Principais (opcional, para visualização)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', alpha=0.6)
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('PCA - Visualização dos Clientes', fontsize=16, color='red')
plt.xticks(fontsize=10, color='red')
plt.yticks(fontsize=10, color='red')
plt.grid(True, linestyle='--', alpha=0.5, color='gray')
plt.gca().set_facecolor('black')
plt.colorbar(scatter, label='Churn (0 = Não, 1 = Sim)')
plt.show()

# Modelagem com regressão logística
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Avaliação do Modelo
print(classification_report(y_test, y_pred))
print('AUC-ROC:', roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

# Curva ROC
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='Curva ROC', color='red')
plt.plot([0, 1], [0, 1], 'k--', label='Classificação Aleatória')
plt.xlabel('Taxa de Falsos Positivos', fontsize=12, color='red')
plt.ylabel('Taxa de Verdadeiros Positivos', fontsize=12, color='red')
plt.title('Curva ROC', fontsize=16, color='red')
plt.xticks(fontsize=10, color='red')
plt.yticks(fontsize=10, color='red')
plt.grid(axis='y', linestyle='--', alpha=0.7, color='gray')
plt.gca().set_facecolor('black')
plt.legend(facecolor='black', framealpha=1, fontsize=10)
plt.show()

# Gráfico de Desempenho (Rosca)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

metrics = {'Precisão': precision, 'Revocação': recall, 'F1-Score': f1}
values = list(metrics.values())
labels = list(metrics.keys())
colors = ['darkred', 'white', 'lightcoral']

plt.figure(figsize=(8, 6))
plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, wedgeprops=dict(width=0.3, edgecolor='darkred'))
plt.title('Desempenho do Modelo', fontsize=16, color='red')
plt.gca().set_facecolor('black')
plt.show()
