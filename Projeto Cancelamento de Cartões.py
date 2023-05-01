#!/usr/bin/env python
# coding: utf-8

# # Desafio: 
# 
# Você trabalha em uma grande empresa de Cartão de Crédito e o diretor da empresa percebeu que o número de clientes que cancelam seus cartões tem aumentado significativamente, causando prejuízos enormes para a empresa
# 
# O que fazer para evitar isso? Como saber as pessoas que têm maior tendência a cancelar o cartão?
# 
# # O que temos:
# 
# Temos 1 base de dados com informações dos clientes, tanto clientes atuais quanto clientes que cancelaram o cartão
# 
# Download da Base de Dados: https://drive.google.com/file/d/1c0F7BDwvOZ9NnIj3tUANuvmp_jRYsh31/view?usp=sharing
# 
# Referência: https://www.kaggle.com/sakshigoyal7/credit-card-customers

# ### Importando a base de Dados

# In[1]:


import pandas as pd

clientes_df = pd.read_csv(r"ClientesBanco.csv", encoding='latin1')
clientes_df = clientes_df.drop('CLIENTNUM', axis=1)
display(clientes_df)


# ### Tratamentos e visão geral dos dados

# In[2]:


clientes_df = clientes_df.dropna()
display(clientes_df.info())
display(clientes_df.describe())


# ### Agora vamos analisar a base, como estão divididos os clientes e os cancelados 

# In[3]:


display(clientes_df['Categoria'].value_counts())
display(clientes_df['Categoria'].value_counts(normalize=True))


# ### Temos vários "approaches" possíveis para resolver o problema.
# 
# - Podemos visualizar como as categorias estão divididas entre Clientes e Cancelados
# 
# - Podemos olhar apenas os Cancelados e olhar como eles estão divididos por categoria, para identificar oportunidades

# In[4]:


import plotly.express as px

# para edições nos gráficos: https://plotly.com/python/histograms/

def grafico_coluna_categoria(coluna, tabela):
  fig = px.histogram(tabela, x=coluna, color='Categoria')
  fig.show()

for coluna in clientes_df:
  grafico_coluna_categoria(coluna, clientes_df)


# # O que conseguimos perceber analisando esses gráficos?
# 
# 1 - Olhando Taxa de Utilização e Qtde de Transações nos Últimos 12m vemos que quem utiliza menos o cartão tende a cancelar mais (se a pessoa tiver mais de 80 transações nos últimos 12 meses, a chance dele cancelar cai bizarramente)
# 
# (O Valor parece ter algo parecido)
# 
# 2 - Parece que quanto mais vezes a pessoa entrou em contato com a gente, mais ela tende a cancelar. Será que esses contatos são problemas/dificuldades de uso que não estão sendo resolvidos?
# 
# 3 - Quase a totalidade dos clientes que cancelam são da categoria Blue, talvez a categoria não seja vantajosa para o cliente. É perfil do cliente? Como estão os concorrentes nessa mesma categoria?

# # Indo mais a fundo
# 
# Podemos agora combinar as features dos nossos palpites para onde deve ser o nosso foco, seguindo a lógica 80-20
# 
# Qual a menor quantidade de problemas que conseguimos atacar que vão gerar o maior resultado?
# 
# Temos quase a totalidade dos Cancelados na bandeira "Blue", então já sabemos que nosso esforço deve ser focado ali
# 
# Mas vamos agora olhar mais a fundo:
#   - Quantidade de Transações nos últimos 12m (ou valor das transações)
#   - Quantidade de Contatos

# In[5]:


cancelados_df = clientes_df.loc[clientes_df['Categoria'] == 'Cancelado', :]

fig = px.histogram(cancelados_df, x='Qtde Transacoes 12m', nbins=5)
fig.show()

fig = px.histogram(cancelados_df, x="Valor Transacoes 12m", nbins=7)
fig.show()


# In[6]:


criticos_df = cancelados_df.loc[cancelados_df['Qtde Transacoes 12m'] < 60, :]
fig = px.histogram(criticos_df, x='Contatos 12m')
fig.show()


# In[7]:


qtde_ultra_criticos = len(criticos_df.loc[criticos_df['Contatos 12m'] > 2, :])

percentual_criticos = qtde_ultra_criticos / len(cancelados_df)
print(f'{percentual_criticos:.1%}')


# Comparação com um projeto usando Machine Learning: https://www.kaggle.com/alpertml/credit-card-customers-eda-ml-97-5-accuracy

# # Outro approach:
# 
# Vamos ver como os Cancelados estão divididos ao todo

# In[8]:


for coluna in cancelados_df:
  grafico_coluna_categoria(coluna, cancelados_df)

