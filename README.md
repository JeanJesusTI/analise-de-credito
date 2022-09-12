# Análise de Crédito
Neste repositório está armazenado um projeto onde iremos realizar uma análise do resultado de 3 algoritimos de machine learning aplicados a um dataset  com dados referentes a liberação de emprestimos. O objetivo é treinar os algoritimos para prever se uma pessoa vai ou não pagar um empréstimo e analisar os resultados

### 1° Importação das bibliotecas necessárias:

```python
  import pandas as pd
  import numpy as np
  import statistics
  from scipy import stats
  from sklearn.model_selection import train_test_split
  from sklearn.naive_bayes import GaussianNB
  from sklearn.linear_model import LogisticRegression
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.metrics import accuracy_score
```

### 2° iremos realizar a leitura do dataset utilizando o pandas
```
dataset = pd.read_csv('credit_data.csv')
```

## Pré-processamento dos dados
Precisaremos remover os dados missing em nosso dataset, para isso, utilizaremos o numpy

```
# Removendo os valores missing
dataset.dropna(inplace=True)
# Verificando se os valores missing foram removidos
dataset.isna().sum()
```
