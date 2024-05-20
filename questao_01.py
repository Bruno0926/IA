# Conversão de Nominal para Numérico
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Carregar os dados
data = pd.read_csv('breast-cancer.csv')

# Supondo que há colunas nominais
label_encoder = LabelEncoder()
onehot_encoder = OneHotEncoder()

for column in data.columns:
    if data[column].dtype == 'object':
        if len(data[column].unique()) == 2:
            data[column] = label_encoder.fit_transform(data[column])
        else:
            encoded = onehot_encoder.fit_transform(data[column].values.reshape(-1,1)).toarray()
            df_encoded = pd.DataFrame(encoded, columns=[f"{column}_{int(i)}" for i in range(encoded.shape[1])])
            data = pd.concat([data.drop(column, axis=1), df_encoded], axis=1)

# Identificação de Outliers
from scipy import stats

z_scores = stats.zscore(data.select_dtypes(include=['float64', 'int64']))
abs_z_scores = abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
data = data[filtered_entries]

# Normalização
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data[data.columns] = scaler.fit_transform(data[data.columns])

# Balanceamento
from imblearn.over_sampling import SMOTE

X = data.drop('target', axis=1)
y = data['target']

smote = SMOTE()
X_res, y_res = smote.fit_resample(X, y)

# Eliminação de Redundância
corr_matrix = data.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
data = data.drop(to_drop, axis=1)
