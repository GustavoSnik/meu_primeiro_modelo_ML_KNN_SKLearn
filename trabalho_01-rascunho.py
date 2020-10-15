#===============================================================================
# TRABALHO 01 - classificador para apoio à decisão de aprovação de crédito.
#===============================================================================

#-------------------------------------------------------------------------------
# Importar bibliotecas
#-------------------------------------------------------------------------------

import pandas as pd
import numpy as np

#-------------------------------------------------------------------------------
# Ler o arquivo CSV com os dados do conjunto de treinamento
#-------------------------------------------------------------------------------

dados = pd.read_csv('conjunto_de_treinamento.csv')    

dados.describe()

dados.mean()
 
dados.median()

#desvio padrão

dados.std()

#-------------------------------------------------------------------------------
# APAGANDO VALORES NÃO NUMÉRICOS
#-------------------------------------------------------------------------------

'''dados.drop('forma_envio_solicitacao', 1, inplace=True)
dados.drop('sexo', 1, inplace=True)
dados.drop('estado_onde_nasceu', 1, inplace=True)
dados.drop('estado_onde_reside', 1, inplace=True)
dados.drop('possui_telefone_residencial', 1, inplace=True)
dados.drop('possui_telefone_celular', 1, inplace=True)
dados.drop('vinculo_formal_com_empresa', 1, inplace=True)
dados.drop('possui_telefone_trabalho', 1, inplace=True)
dados.drop('estado_onde_trabalha', 1, inplace=True)'''

#-------------------------------------------------------------------------------
# tratar valores nulos
#-------------------------------------------------------------------------------

#Imputer coloca valores de entrada, sendo responsável pela correção de valores faltantes








'''from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')


#trabalha direto com o dataframe dados
imputer = imputer.fit(previsores[:, 0:32])
previsores[:,0:32] = imputer.transform(previsores[:,0:32])'''







#função loc para localizar  

#dados.loc[dados['idade'] < 18]


#dados.describe()

#-------------------------------------------------------------------------------
# apagar coluna
#-------------------------------------------------------------------------------

#dados.drop('idade', 1, income=True)




#-------------------------------------------------------------------------------
# apagar somente os registros com problemas 
#-------------------------------------------------------------------------------

#dados.drop(dados[dados.idade < 0 ].index, inplace=True)

#-------------------------------------------------------------------------------
# preencher os valores com a média
#-------------------------------------------------------------------------------

# 1° ver a média
dados.mean()

# ver a média somente da idade
dados['idade'].mean()

#igualar a média os valores invalidos
dados['idade'][dados.idade > 0].mean()

#-------------------------------------------------------------------------------
# localizar e atualizar (caso houvesse idade menor que zero)
#-------------------------------------------------------------------------------

# dados.loc[dados.idade < 0, 'idade'] = 43.92

#-------------------------------------------------------------------------------
# TRATAMENTO DE VALORES FALTANTES
#-------------------------------------------------------------------------------

pd.isnull(dados['idade'])


#dados.loc[pd.isnull(dados)]

#Localizando valores de idades nulos

dados.loc[pd.isnull(dados['idade'])]

#-------------------------------------------------------------------------------
#UMA VARIÁVEL VAI ARMAZENAR OS ATRIBUTOS PREVISORES E UM ATRIBUTO CLASSE
#-------------------------------------------------------------------------------

# POR AS VARIÁVEIS PREVISORES (iloc faz a divisão)
# O primeiro parÂmetro indica as linhas (o id não é útil por ser único e não haver padrão)
previsores = dados.iloc[:, 1:32].values

classe = dados.iloc[:, 32].values



#-------------------------------------------------------------------------------
# TRATAR DADOS FALTANTES
#
#Substituir os dados pela média ou mediana(verificar melhor opção) 
#-------------------------------------------------------------------------------


#dados2 = dados.dropna() #ELIMINA LINHA DE VALORES NULOS

#enulo = dados.isnull()  #VERIFICA QUAIS SÃO NULOS

#faltante = dados.isnull().sum() #LISTA COLUNAS NULAS

#faltantes_percentual = (dados.isnull().sum() / len(dados['id_solicitante']))*100 #percentual dos Dados Faltantes

dados['tipo_residencia'].fillna(dados['tipo_residencia'].mean(),inplace = True)
dados['meses_na_residencia'].fillna(dados['meses_na_residencia'].mean(),inplace = True)
dados['profissao'].fillna(dados['profissao'].mean(),inplace = True)
dados['ocupacao'].fillna(dados['ocupacao'].mean(),inplace = True)
dados['profissao_companheiro'].fillna(dados['profissao_companheiro'].mean(),inplace = True)
dados['grau_instrucao_companheiro'].fillna(dados['grau_instrucao_companheiro'].mean(),inplace = True)
dados['local_onde_reside'].fillna(dados['local_onde_reside'].mean(),inplace = True)
dados['local_onde_trabalha'].fillna(dados['local_onde_trabalha'].mean(),inplace = True)


#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Selecionar os atributos que serão utilizados pelo classificador
#-------------------------------------------------------------------------------

atributos_selecionados = ['produto_solicitado',
                          'dia_vencimento',
                          'tipo_endereco',
                          'idade',                       
                          'estado_civil',
                          'qtde_dependentes',
                          'grau_instrucao',
                          'nacionalidade',
                          'possui_telefone_residencial',
                          'tipo_residencia',
                          'meses_na_residencia',
                          'possui_telefone_celular',
                          'possui_email',
                          'renda_mensal_regular',
                          'renda_extra',
                          'possui_cartao_visa',
                          'possui_cartao_mastercard',
                          'possui_cartao_diners',
                          'possui_cartao_amex',
                          'possui_outros_cartoes',
                          'qtde_contas_bancarias',
                          'qtde_contas_bancarias_especiais',
                          'valor_patrimonio_pessoal',
                          'possui_carro',
                          'vinculo_formal_com_empresa',
                          'possui_telefone_trabalho',
                          'meses_no_trabalho',
                          'profissao',
                          'ocupacao', 
                          'profissao_companheiro',
                          'grau_instrucao_companheiro',
                          'local_onde_reside',
                          'local_onde_trabalha',                         
                          #'sexo_ ',
                          'sexo_F',
                          'sexo_M',
                          #'sexo_N']

dados = dados[atributos_selecionados]

#-------------------------------------------------------------------------------
# Embaralhar o conjunto de dados para garantir que a divisão entre os dados de
# treino e os dados de teste esteja isenta de qualquer viés de seleção
#-------------------------------------------------------------------------------

dados_embaralhados = dados.sample(frac=1,random_state=12345)

#-------------------------------------------------------------------------------
# Criar os arrays X e Y separando atributos e alvo
#-------------------------------------------------------------------------------

x = dados_embaralhados.loc[:,dados_embaralhados.columns!='inadimplente'].values
y = dados_embaralhados.loc[:,dados_embaralhados.columns=='inadimplente'].values

#-------------------------------------------------------------------------------
# Separar X e Y em conjunto de treino e conjunto de teste
#-------------------------------------------------------------------------------

q = 30000  # qtde de amostras selecionadas para treinamento

# conjunto de treino

x_treino = x[:q,:]
y_treino = y[:q].ravel() #ver função ravel

# conjunto de teste

x_teste = x[q:,:]
y_teste = y[q:].ravel()

#-------------------------------------------------------------------------------
# Treinar um classificador KNN com o conjunto de treino
#-------------------------------------------------------------------------------

classificador = KNeighborsClassifier(n_neighbors=5)

classificador = classificador.fit(x_treino,y_treino)

#-------------------------------------------------------------------------------
# Obter as respostas do classificador no mesmo conjunto onde foi treinado
#-------------------------------------------------------------------------------

y_resposta_treino = classificador.predict(x_treino)

#-------------------------------------------------------------------------------
# Obter as respostas do classificador no conjunto de teste
#-------------------------------------------------------------------------------

y_resposta_teste = classificador.predict(x_teste)

#-------------------------------------------------------------------------------
# Verificar a acurácia do classificador
#-------------------------------------------------------------------------------

print ("\nDESEMPENHO DENTRO DA AMOSTRA DE TREINO\n")

total   = len(y_treino)
acertos = sum(y_resposta_treino==y_treino)
erros   = sum(y_resposta_treino!=y_treino)

print ("Total de amostras: " , total)
print ("Respostas corretas:" , acertos)
print ("Respostas erradas: " , erros)

acuracia = acertos / total

print ("Acurácia = %.1f %%" % (100*acuracia))

print ("\nDESEMPENHO FORA DA AMOSTRA DE TREINO\n")

total   = len(y_teste)
acertos = sum(y_resposta_teste==y_teste)
erros   = sum(y_resposta_teste!=y_teste)

print ("Total de amostras: " , total)
print ("Respostas corretas:" , acertos)
print ("Respostas erradas: " , erros)

acuracia = acertos / total

print ("Acurácia = %.1f %%" % (100*acuracia))