#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#baixar os dados


# In[1]:


#Download da base de dados MIT-BIH Arrhythmia Database direto do PhisioNet
import wfdb

# Nome do dataset no PhysioNet
database = "mitdb"

# Pasta onde será salvo
target_folder = "./mitbih"

# Faz o download completo
wfdb.dl_database(database, dl_dir=target_folder)

print("Download concluído!")


# In[ ]:


#analisar o que foi baixado


# In[1]:


import pandas as pd
import os
import wfdb

# Pasta onde os arquivos do MIT-BIH estão
data_folder = "mitbih"

# Pega todos os arquivos .dat (cada record tem .dat, .hea e .atr)
records = [os.path.splitext(f)[0] for f in os.listdir(data_folder) if f.endswith(".dat")]

# Remove duplicatas e ordena
records = sorted(set(records))

print("Registros encontrados:", records)

# Pasta para salvar os CSVs
output_folder = "mit-bih-arrhythmia-database-1.0.0"
os.makedirs(output_folder, exist_ok=True)
print("Criação da pasta concluida")


# In[13]:


#anotação dos dados


# In[14]:


for rec_name in records:
    print(f"Processando registro {rec_name}...")

    # Lê o sinal
    record = wfdb.rdrecord(os.path.join(data_folder, rec_name))
    signal = record.p_signal  # array numpy, cada coluna = canal

    # Lê as anotações
    annotation = wfdb.rdann(os.path.join(data_folder, rec_name), 'atr')

    # Cria DataFrame com todas as amostras
    df = pd.DataFrame(signal, columns=[f"channel_{i}" for i in range(signal.shape[1])])
    df["sample #"] = range(len(df))
    df["type"] = "NC"  # default sem anotação

    # Marca os batimentos anotados
    for s, t in zip(annotation.sample, annotation.symbol):
        if s < len(df):
            df.loc[s, "type"] = t

    # Salva CSV
    output_csv = os.path.join(output_folder, f"{rec_name}_labeled.csv")
    df.to_csv(output_csv, index=False)

    print(f"{rec_name}_labeled.csv criado!")


# In[ ]:


#juntar todos os dados em um unico csv


# In[2]:


import pandas as pd
import os

# Pasta CSV
csv_folder = "mit-bih-arrhythmia-database-1.0.0"


# In[3]:


# Lista todos os arquivos .csv da pasta em ordem alfabética
csv_files = sorted(
    [os.path.join(csv_folder, f) for f in os.listdir(csv_folder) if f.endswith(".csv")]
)
print(csv_files)


# In[4]:


# Lê e concatena tudo em um único DataFrame
df_list = []
for file in csv_files:
    temp_df = pd.read_csv(file)

    # adiciona coluna extra para saber de qual registro veio
    record_name = os.path.splitext(os.path.basename(file))[0]
    temp_df["record"] = record_name  

    df_list.append(temp_df)

df_all = pd.concat(df_list, ignore_index=True)

# Salva em um único CSV
df_all.to_csv("mitbih_all_records.csv", index=False)

print("Arquivo único gerado: mitbih_all_records.csv")


# In[6]:


# Contagem de cada tipo
# Carregar a planilha unificada
df = pd.read_csv("mitbih_all_records.csv")
contagem = df["type"].value_counts()

print(contagem)


# In[7]:


df.head(20)


# In[8]:


# Reenumera a coluna 'sample #' a partir de 0
df['sample #'] = range(len(df))


# In[9]:


df.head(20)


# In[10]:


df.to_csv("mitbih_all_records_renumerada.csv", index=False)


# In[19]:


#limpeza de dados


# In[9]:


# Carregar a planilha
df = pd.read_csv("mitbih_all_records.csv")

# Lista de tipos relevantes (batimentos cardíacos)
tipos_relevantes = ["N", "L", "R", "V", "A", "F", "S", "Q", "/", "f", "NC"]

# Filtrar somente esses tipos
df_filtrado = df[df["type"].isin(tipos_relevantes)]

# Salvar em novo arquivo
df_filtrado.to_csv("batimentos_filtrados.csv", index=False)

print("Linhas originais:", len(df))
print("Linhas após filtragem:", len(df_filtrado))


# In[3]:


# Carregar a planilha
df = pd.read_csv("batimentos_filtrados.csv")

# Contagem de cada tipo
contagem = df["type"].value_counts()

print(contagem)


# In[4]:


df.head(20)


# In[6]:


contagem = df["sample #"]


# In[17]:


ultimo_valor = contagem.tail(1).values[0]


# In[18]:


print(ultimo_valor)

