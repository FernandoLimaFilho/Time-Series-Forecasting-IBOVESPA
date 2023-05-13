# Previsão de séries temporais: IBOVESPA

<h1>
  
```python
print("Introdução")
```    
  
</h1>  

Índice Bovespa é o mais importante indicador do desempenho médio das cotações das ações negociadas na B3 - Brasil, Bolsa, Balcão. É formado pelas ações com maior volume negociado nos últimos meses.

Em poucas palavras, o IBOVESPA é o que marca o desempenho da bolsa de valores brasileira.

<p align="center">
 
![ibovespa-1](https://github.com/FernandoLimaFilho/Time-Series-Forecasting-IBOVESPA/assets/93550626/6ce09e4f-136e-442b-a270-1fd2968c186f)
 
<p>
 
  
<h1>

```python
print("Objetivos")
```  
 
</h1>  

* O objetivo desse projeto é realizar uma previsão acurada do IBOVESPA de no mínimo 2 meses. Tal feito foi realizado utilizando a linguagem de programação python
 
<h1>

```python
print("Procedimentos")
```
  
</h1>    
 
## Conjunto de dados original 

O conjunto de dados original foi extraído de: http://www.investing.com/currencies/usd-brl-historical-data

* Data: Data de cotagem
* Último: Última avaliação do IBOVESPA no dia
* Abertura: Primeira avaliação do IBOVESPA no dia
* Máxima: Máxima avaliação do IBOVESPA no dia
* Mínima: Mínima avaliação do IBOVESPA no dia

## Importação das bibliotecas

```bash
"""
1°) Importação do pandas como pd para trabalhar com dados.
"""
import pandas as pd
"""
2°) Importação do numpy como np para trabalhar com matrizes e tudo mais.
"""
import numpy as np
"""
3°) Importação do matplotlib.pyplot como plt para fazer gráficos.
"""
import matplotlib.pyplot as plt
"""
4°) De matplotlib.ticker vamos importar o AutoMinorLocator e o MaxNLocator para trabalhar com os "ticks"
    dos gráficos.
"""
import matplotlib.ticker as mticker
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
"""
5°) De matplotlib.font_manager vamos importar FontProperties para criar fontes de texto.
"""
from matplotlib.font_manager import FontProperties
"""
6°) Importação do seaborn para fazer gráficos
"""
import seaborn as sbn
"""
7°) Importação de pycaret.time_series para trabalhar com séries temporais
"""
from pycaret.time_series import *
from pycaret.internal.pycaret_experiment import TimeSeriesExperiment
from sktime.utils.plotting import plot_series
```

## Chamando os dados

```bash
Dados = pd.read_csv("Dados Históricos - Ibovespa.csv") # Ler os dados da extensão .csv
Dados.drop(["Vol.", "Var%"], axis = 1, inplace = True) # Exclusão de duas colunas desnecessárias
Dados.head(7) # Mostrar 7 linhas
```
 
## Pré-processamento de dados

Vamos verificar, primeiramente, os dtypes, 

```bash
Dados = Dados.replace(",",".", regex = True) # Tudo que é vírgula vira ponto
Dados["Data"] = pd.to_datetime(Dados["Data"], format = "%d.%m.%Y")# Tranformando no formato de data
Dados['Data'] = Dados['Data'].dt.strftime('%Y-%m-%d')
Dados["Último"] = Dados["Último"].astype(float) # Tranformando em float
Dados["Abertura"] = Dados["Abertura"].astype(float)
Dados["Máxima"] = Dados["Máxima"].astype(float)
Dados["Mínima"] = Dados["Mínima"].astype(float)
Dados.dtypes
```

Após isso, vamos ver se existem valores nulos, 

```bash
Valores_nulos_percentual = 100*(Dados.isnull().sum()/len(Dados["Mínima"]))
print(Valores_nulos_percentual)
```

Por fim, adicionemos uma coluna do valor médio do dólar, 

```bash
Dados["Média"] = Dados[["Máxima", "Mínima"]].mean(axis = 1)
```

<h1>

```python
print("Previsão do IBOVESPA")
```
  
</h1>


```bash
Serie_temporal = Dados[["Data", "Média"]] 
Serie_temporal.index = pd.date_range(end = "2023-05-11", periods=1741, freq = "D")
Serie_temporal = Serie_temporal.drop("Data", axis = 1)
Serie_temporal = Serie_temporal[::-1]
Media_correta = []
for i in range(1830):
    Media_correta.append(Serie_temporal["Média"][i])
Serie_temporal = Serie_temporal[::-1]
Serie_temporal["Media_correta"] = Media_correta
Serie_temporal.drop(["Média"], axis = 1, inplace = True)
Serie_temporal
```

Vamos criar um setup.

```bash
setup(Serie_temporal, fh=120, fold=13, seasonal_period="D", n_jobs = -1, use_gpu = True); # Criando um setup
Compare = compare_models(exclude=['auto_arima']) # Comparar modelos
theta = create_model("theta") # Criar o melhor modelo
final = finalize_model(theta) # finalizar o modelo
```

Realizaremos as predições:

```bash
pred = predict_model(final, fh = 60)
pred = pd.DataFrame(pred, columns = ["Data", "Media_correta"]) # Transformando em DataFrame
pred["Data"] = pred.index.to_timestamp()
pred
```

Plot do gráfico:

```bash
"""
Criação da primeira fonte de texto para colocar como fonte dos labels
"""
font1 = {"family": "serif", "weight": "bold", "color": "gray", "size": 14}
"""
Criação da segunda fonte de texto para colocar como fonte da legenda
"""
font2 = FontProperties(family = "serif",
                       weight = "bold",
                       size = 14)
"""
Cria um "lugar" com size (9, 7) para alocar a figura
"""
fig, axs = plt.subplots(figsize = (14, 7))
"Plot do gráfico"
axs.plot(pred["Data"],
         pred["Media_correta"],
         color = "blue",
         linewidth = 1.5,
         label = "Previsão (2023-05-12 até 2023-07-10)")
axs.grid(False)
"""
Definindo a "grossura" e a cor do eixos
"""
for axis in ["left", "right", "top", "bottom"]:
    axs.spines[axis].set_linewidth(2)
    axs.spines[axis].set_color("gray")
"""
Trabalha com os ticks do gráfico
"""    
axs.xaxis.set_minor_locator(AutoMinorLocator())
axs.yaxis.set_minor_locator(AutoMinorLocator())
axs.tick_params(axis = "both", direction = "in", labelcolor = "gray", labelsize = 14, left = True, bottom = True, top = True, right = True)
axs.tick_params(which = "major", direction = "in", color = "gray", length = 5.4, width = 2.5, left = True, bottom = False, top = False, right = True)
axs.tick_params(which = "minor", direction = "in", color = "gray", length=4, width = 2, left = True, bottom = True, top = True, right = True)
"""
Descrição para cada eixo
"""
axs.set_xlabel("Data", fontdict = font1)
axs.set_ylabel("USD/BRL", fontdict = font1)
"""
plt.rcParams["axes.labelweight"] = "bold" mostra em negrito os números nos eixos.
"""
plt.rcParams["axes.labelweight"] = "bold"
plt.legend(frameon = False, prop = font2, labelcolor = "gray")
"""
Definindo um fundo branco para a imagem
"""
fig.patch.set_facecolor("white")
Cor_fundo = plt.gca()
Cor_fundo.set_facecolor("white")
Cor_fundo.patch.set_alpha(1)
"""
Mostrar o gráfico
"""
plt.show()
```

<p align="center">
 
![download (31)](https://github.com/FernandoLimaFilho/Time-Series-Forecasting-IBOVESPA/assets/93550626/79b8cf14-7632-4de5-a58a-d8fdd2c555a2)
 
<p>
