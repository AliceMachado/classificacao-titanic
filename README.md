# classificacao-titanic
Exercício Inteligência Artificial, com o desafio do Titanic do Kaggle

Para melhor visualização, utilize o modo 'Code' do arquivo README!

Introdução:
Este projeto é uma análise dos dados do Titanic disponibilizado no Kaggle. O objetivo é prever a sobrevivência dos passageiros usando um modelo de classificação k-NN.

-> Carregamento e pré-processamento dos dados:

#Importe e download dos arquivos
from google.colab import files
arquivos = files.upload()

#Importe do Pandas e io para leitura dos arquivos .CSV
import pandas as pd
import io

#Leitura do arquivo train.csv
df = pd.read_csv(io.BytesIO(arquivos['train.csv']))

#Ver as primeiras linhas do arquivo
df.head()

#Ver a estrutura das colunas do arquivo
df.info()

#Descrever estatísticas básicas do arquivo
df.describe()

#Identificando valores nulos
df.isnull().sum()

#Preencher valores nulos na coluna 'Age' com a média
df['Age'] = df['Age'].fillna(df['Age'].mean())

#Remover linhas com valores nulos na coluna 'Embarked'
df = df.dropna(subset=['Embarked'])

#Selecionando as colunas para variáveis
variaveis = ['Pclass', 'Sex', 'Age', 'Fare', 'Survived']
df = df[variaveis]

#Convertendo a variável 'Sex' em números
df = pd.get_dummies(df, columns=['Sex'], drop_first=True)

#Dividindo os dados, 'Sex_male' recebe o valor 1 para homens e 0 para mulheres
X = df[['Pclass', 'Sex_male', 'Age', 'Fare']]
y = df['Survived']

-> Implementação do algoritmo k-NN:

#Importe da biblioteca scikit-learn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#Dividindo os dados entre treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Criar e treinar o modelo k-NN com k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

#Usando o modelo treinado para prever as classes no conjunto de teste
y_pred = knn.predict(X_test)

#Importe da biblioteca scikit-learn para avaliar a acurácia do modelo
from sklearn.metrics import accuracy_score

#Calcular e mostrar a acurácia das previsões realizadas
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy:.2f}')

#Importe da biblioteca scikit-learn para avaliar o desempenho do modelo
from sklearn.metrics import confusion_matrix

#Calcular e mostrar o desempenho ao comparar as classificações corretas e incorretas
cm = confusion_matrix(y_test, y_pred)
print(cm)

-> Avaliação de desempenho (acurácia e matriz de confusão), observações sobre os resultados e o que pode ser melhorado:

Resultados:
Acurácia: 0.70
Matriz de Confusão:
[[139  28]
 [ 51  49]]

Nas observações sobre o desempenho do modelo, a acurácia do modelo foi de 70%, o que significa que 70% das previsões do modelo estavam corretas. Isso nos dá uma boa noção de que o modelo está fazendo previsões melhores do que a sorte, mas ainda há bastante espaço para melhorar.

Agora sobre a matriz de confusão, ela nos retornou o seguinte:

Interpretando este resultado:
Positivos verdadeiros(49): O modelo previu corretamente que 49 pessoas sobreviveram.
Negativos verdadeiros(139): O modelo previu corretamente que 139 pessoas não sobreviveram.

Positivos falsos(28): O modelo previu incorretamente que 28 pessoas sobreviveram, mas na verdade não sobreviveram.
Negativos falsos(51): O modelo previu incorretamente que 51 pessoas não sobreviveram, mas na verdade sobreviveram.

Sobre as possíveis melhorias:
	- Testar diferentes valores de k: As variações no valor de k podem ter um grande impacto no desempenho do modelo. Então ao testar diferentes valores, podemos identificar um que forneça uma melhor acurácia.
	- Criar novas variáveis que possam ser relevantes para o modelo, e normalizando os dados para garantir que todas as variáveis tenham a mesma escala pode ajudar a melhorar o desempenho.
