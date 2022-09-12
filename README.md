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

### 2° Iremos realizar a leitura do dataset utilizando o pandas
```python
dataset = pd.read_csv('credit_data.csv')
```

## Pré-processamento dos dados
Precisaremos remover os dados missing em nosso dataset, para isso, utilizaremos o numpy

```python
# Removendo os valores missing
dataset.dropna(inplace=True)
# Verificando se os valores missing foram removidos
dataset.isna().sum()
```

### Divisão dos dados de treino e teste
Outra parte importante, é dividir os dados em dados de treino e teste para realizarmos o treinamento do algoritimo e realizar previsões; para isso iremos fazer as seguintes divisões:
X: Variáveis preditoras (Que irão auxiliar o algoritimo a realizar previsões)
y: Variável target (Armazenarão dados que queremos prever)

OBS: Não utilizaremos os campos de ID's pois não são relevantes para um algoritmo de machine learning, visto que servem apenas como chave primária para cada pessoa presente no dataset.

```python
X = dataset.iloc[:, 1:4].values
y = dataset.iloc[:,4].values
```

## Treinamento dos modelos de Machine Learning
Nesta parte do projeto, iremos utilizar uma técnica bastante recomendada pela comunidade cientifica para obtermos os resultados, que consistem em realizar 30 treinamentos diferentes realizando uma variação nos dados, a fim de conseguirmos analisar a performance dos algoritimos baseados na quantidade dos testes realizados e seus resultados para cada um deles.
Os resultados serão armazenados em listas para que possamos extrair posteriormente as medidas de disperção, que nos mostrará qual melhor algoritmo com base na análise estatistica.

```python
  from sklearn import naive_bayes

  resultados_naive_bayes = []
  resultados_logistica = []
  resultados_forest = []

  for i in range(30):
    X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X,y, test_size=0.2, 
                                                                      stratify=y,
                                                                      random_state=i)

    # Algoritimo NaiveBayes
    naive_bayes = GaussianNB()
    naive_bayes.fit(X_treinamento, y_treinamento)
    resultados_naive_bayes.append(accuracy_score(y_teste, naive_bayes.predict(X_teste)))

    # Algoritimo Regressão Logistica
    logistic = LogisticRegression()
    logistic.fit(X_treinamento, y_treinamento)
    resultados_logistica.append(accuracy_score(y_teste, logistic.predict(X_teste)))

    # Algoritimo RandomForest
    random_forest = RandomForestClassifier()
    random_forest.fit(X_treinamento, y_treinamento)
    resultados_forest.append(accuracy_score(y_teste, random_forest.predict(X_teste)))

```

## Análise dos Resultados
Como os resultados foram armazenados em uma lista, precisaremos realizar uma conversão para array, com isso, conseguiremos analisar as medidas de formas mas simples e objetivas, para essa conversão, utilizaremos no Numpy para sobrescrever as variáveis conforme codigo abaixo:

```python
  resultados_naive_bayes = np.array(resultados_naive_bayes)
  resultados_logistica = np.array(resultados_logistica)
  resultados_forest = np.array(resultados_forest)
```

### Calculando Média dos resultados:
```python
resultados_naive_bayes.mean(), resultados_logistica.mean(), resultados_forest.mean()
# Resultados: (0.92425, 0.9145, 0.98475)
```

### Calculando a Moda dos resultados:
```python
  stats.mode(resultados_naive_bayes), stats.mode(resultados_logistica), stats.mode(resultados_forest)
```

<p align="center">
  Resultado: <br/>
  <img src="https://user-images.githubusercontent.com/31626353/189756576-5e216e0a-0875-4849-aad0-2b4c236cdc86.png">
</p>

### Calculando a Mediana dos resultados:
```python
  np.median(resultados_naive_bayes), np.median(resultados_logistica), np.median(resultados_forest)
```


<p align="center">
  Resultados: <br/>
  <img src="https://user-images.githubusercontent.com/31626353/189756939-16848ce9-be0a-4467-a971-96a4d37e8937.png">
</p>

### Calculando a Variância dos resultados:
Como os dados estão em notação cientifica, visualmente é difícil definir qual dos algoritimos possui a menor variância, para isso, utilizaremos uma função do numpy passando os resultados para que ele nos retorne o algoritimo que retornou a menor variância.

```python
  np.min([8.756250000000001e-05, 0.00020933333333333337, 3.139583333333343e-05])
```
<p align="center">
  Resultados: <br/>
  <img src="https://user-images.githubusercontent.com/31626353/189759050-801f07eb-c574-4697-a786-5f2e16a173dd.png">
</p>


Podemos observar que o algoritmo RandomForestClassifier é o modelo que tem a menor variância; isso indica que os resultados foram consistentes em relação a média, e que os dados não estão variando muito dentre os 30 testes realizados.




### Calculando o Desvio padrão dos resultados:
Para calcular-mos o desvio padrão, iremos utilizar um método do numpy chamado std
```python
  np.std(resultados_naive_bayes), np.std(resultados_logistica), np.std(resultados_forest)
```
<p align="center">
  Resultado: <br/>
  <img src="https://user-images.githubusercontent.com/31626353/189758687-4701f79a-2ad6-4d1a-88ee-986b30eae2cf.png">
</p>

### Calculando o Coeficiente de variação dos resultados:
```python
  stats.variation(resultados_naive_bayes) * 100, stats.variation(resultados_logistica) * 100, stats.variation(resultados_forest) * 100
```
<p align="center">
  Resultado: <br/>
  <img src="https://user-images.githubusercontent.com/31626353/189759403-90e7e5b3-2438-4f57-b808-557250aecba0.png">
</p>

Podemos ver que os algoritimos obtiveram respectivamente as seguintes variações:<br/>
GaussianNB: **10%**. <br/>
LogisticRegression: **10.5%**.  <br/>
RandomForestClassifier: **0.5%**. <br/>

## Conclusão:
<p align="center">
Podemos concluir que o algoritimo que teve um melhor resultado foi o algoritimo **RandomForestClassifier**, devido a análise individual das métricas de cada um deles. A conclusão não se baseia no resultado de acuracidade (que coincidentemente foi o maior), pois os outros algoritimos poderiam ter resultados mais elevados, porém, a variancia dos resultados poderiam fazer com que os resultados não fossem tão acertivos, fazendo com que esse algoritimo tivesse resultados discrepantes, causando um equivoco se levarmos em consideração apenas sua acuracidade.
</p>
