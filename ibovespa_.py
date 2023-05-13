#!/usr/bin/env python
# coding: utf-8

# # ${\color{orange}{\text{Machine Learning nas finanças}}}$

# # ${\color{orange}{\text{IBOVESPA}}}$

# # ${\color{orange}{\text{1. Importação das bibliotecas}}}$

# In[52]:


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


# # $\color{orange}{\text{2. Trazendo dados para o python}}$

# In[53]:


Dados = pd.read_csv("Dados Históricos - Ibovespa.csv") # Ler os dados da extensão .csv
Dados.drop(["Vol.", "Var%"], axis = 1, inplace = True) # Exclusão de duas colunas desnecessárias
Dados.head(7) # Mostrar 7 linhas


# In[54]:


Dados.head(20)


# In[55]:


Dados.columns


# 
# Data: Data de cotagem
# 
# Último: Última avaliação do Dólar no dia
# 
# Abertura: Primeira avaliação do Dólar no dia
# 
# Máxima: Máxima avaliação do Dólar no dia
# 
# Mínima: Mínima avaliação do Dólar no dia

# # $\color{orange}{\text{3. Pré-Processamento de dados}}$

# ## $\color{orange}{\text{3.1 Dtypes}}$

# In[56]:


Dados.dtypes


# In[57]:


Dados = Dados.replace(",",".", regex = True) # Tudo que é vírgula vira ponto
Dados["Data"] = pd.to_datetime(Dados["Data"], format = "%d.%m.%Y")# Tranformando no formato de data
Dados['Data'] = Dados['Data'].dt.strftime('%Y-%m-%d')
Dados["Último"] = Dados["Último"].astype(float) # Tranformando em float
Dados["Abertura"] = Dados["Abertura"].astype(float)
Dados["Máxima"] = Dados["Máxima"].astype(float)
Dados["Mínima"] = Dados["Mínima"].astype(float)
Dados.dtypes


# ## $\color{orange}{\text{3.2 Valores nulos}}$

# In[58]:


Valores_nulos_percentual = 100*(Dados.isnull().sum()/len(Dados["Mínima"]))
print(Valores_nulos_percentual)


# Não há nenhum valor nulo no dataset!

# ## $\color{orange}{\text{3.3 Valor médio do Dólar no dia}}$

# In[59]:


Dados["Média"] = Dados[["Máxima", "Mínima"]].mean(axis = 1) # Tirando uma média entre duas colunas
Dados.head(5)


# ## $\color{orange}{\text{3.4 Análise de dados}}$

# In[60]:


Dados.shape


# In[61]:


datatoexcel = pd.ExcelWriter('IBOVESPA.xlsx')
Dados.to_excel(datatoexcel)
datatoexcel.save()
print('DataFrame is written to Excel File successfully.')


# ## $\color{orange}{\text{4. Previsão da série temporal de câmbio}}$

# In[62]:


Serie_temporal = Dados[["Data", "Média"]] 
Serie_temporal.index = pd.date_range(end = "2023-05-11", periods=1741, freq = "D")
Serie_temporal = Serie_temporal.drop("Data", axis = 1)


# In[63]:


Serie_temporal = Serie_temporal[::-1]


# In[64]:


Media_correta = []
for i in range(1741):
    Media_correta.append(Serie_temporal["Média"][i])
Serie_temporal = Serie_temporal[::-1]


# In[65]:


Serie_temporal["Media_correta"] = Media_correta
Serie_temporal.drop(["Média"], axis = 1, inplace = True)
Serie_temporal


# In[66]:


setup(Serie_temporal, fh=120, fold=13, seasonal_period="D", n_jobs = -1, use_gpu = True); # Criando um setup


# In[67]:


Compare = compare_models(exclude=['auto_arima']) # Comparar modelos


# In[68]:


omp_cds_dt = create_model("omp_cds_dt") # Criar o melhor modelo


# In[69]:


final = finalize_model(omp_cds_dt) # finalizar o modelo


# In[70]:


"""
Predições
"""
pred = predict_model(final, fh = 60)
pred = pd.DataFrame(pred, columns = ["Data", "Media_correta"]) # Transformando em DataFrame
pred["Data"] = pred.index.to_timestamp()
pred


# In[71]:


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
         color = "orange",
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
axs.set_ylabel("IBOVESPA", fontdict = font1)
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

