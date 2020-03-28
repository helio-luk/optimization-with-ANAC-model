import pandas as pd
import csv
from numpy import array
import numpy as np
import math
import random
import warnings
from random import randint

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

def respeitaDisponibilidade(a0, disponibilidade, demanda, arcos,inspecoes):
    matriz_disp = a0.copy().astype(float)
    disp_aux = disponibilidade['NOME'].values
    demanda_aux = demanda['ATIVIDADE'].values
    servidor_uf = np.zeros((len(a0[:,0])), dtype=object)
    missao_icao = np.zeros((len(a0[0,:])), dtype=object)
       
    #anda pelo eixo y, ou seja, vai de servidor em servidor
    for i in range (0, len(a0[:,0])):
        indices_missoes = np.where(a0[i,:] == 1)[0]
        origem = disponibilidade.where(disponibilidade['NOME'].isin([disp_aux[i]]) == True).dropna()            
        origem = origem['UF'].values[0]        
        servidor_uf[i] = origem
        
        for j in range (0, len(indices_missoes)):
            destino = demanda['AEROPORTO'].values[indices_missoes[j]]
            tempo_total = inspecoes['DURACAO'].loc[inspecoes['GRUPO_A'] == demanda_aux[indices_missoes[j]]].values[0]
            tempo_total = tempo_total + (2*(arcos['TEMPO'].loc[(arcos['ORIGEM_ARCO'] == origem) & (arcos['DESTINO_ARCO'] == destino)].values[0]))
            matriz_disp[i,indices_missoes[j]] = tempo_total
            missao_icao[indices_missoes[j]] = destino
            
        missao_icao = np.zeros((len(a0[0,:])), dtype=object)
        
    for i in range (0, len(a0[:,0])):
        #ALTERAR O 5.0 PARA A DISPONIBILIDADE DO SERIVDOR
        if(np.sum(matriz_disp[i,:]) > 5.0):
            return False

    return True

#calcula custo da matriz
def calculaCusto(a0, disponibilidade, demanda, arcos):
    disp_aux = disponibilidade['NOME'].values
    custo_total = 0
    
    for i in range (0, len(a0[0,:])):
        indice_aux = np.where(a0[:,i] == 1)[0][0]

        origem = disponibilidade.where(disponibilidade['NOME'].isin([disp_aux[indice_aux]]) == True).dropna()            
        origem = origem['UF'].values[0]
        destino = demanda['AEROPORTO'].values[i]                

        valor = arcos.where(arcos['ORIGEM_ARCO'].isin([origem]) == True).dropna()
        valor = valor.where(valor['DESTINO_ARCO'].isin([destino]) == True).dropna()
        valor = valor['CUSTO'].values
        valor = valor[0]
        custo_total = custo_total + valor   
    return custo_total
    

#direcao: -1 desce, 1 sobe
#a0:        matriz com a solução possivel
#FO:       matriz gabarito 
def salta(a0, direcao,FO,y_inicio, y_fim):
    a_aux = a0.copy()
    
    for i in range (0, len(FO[0,:])):
        #fazer o recorte do salto em cima dos servidores_aptos tambem
        servidores_aptos = np.where(FO[:,i] == 1)[0]        
        servidores_aptos = servidores_aptos[(servidores_aptos >= y_inicio) &  (servidores_aptos < y_fim)]
        try:
            
            indice_atual = np.where(a0[:,i] == 1)[0][0]            
            if((indice_atual >= y_inicio) & (indice_atual < y_fim)):                
                servidor_salto = np.where(servidores_aptos == indice_atual)[0][0]
            
                if((servidor_salto - direcao) == len(servidores_aptos)):                
                    novo_indice = servidores_aptos[0]
                else:
                    novo_indice = servidores_aptos[servidor_salto - direcao]

                a_aux[indice_atual,i] = 0
                a_aux[novo_indice,i] = 1
        except:
            l = True
    return  a_aux
    

#função que conserva da parte deslocada e adiciona a parte 
def mascara(a0, a1, y_inicio, y_fim):
    a_aux = np.zeros((len(a0[:,0]),len(a0[0,:])), dtype=int)    
    a_aux[y_inicio:y_fim,:] = np.absolute(~a_aux[y_inicio:y_fim,:])
    a_aux = np.logical_and(a_aux,a0).astype(int)    
    
    a_aux2 = np.ones((len(a0[:,0]),len(a0[0,:])), dtype=int)    
    a_aux2[y_inicio:y_fim,:] = np.logical_not(a_aux2[y_inicio:y_fim,:]).astype(int)    
    a_aux2 = np.logical_and(a_aux2,a1).astype(int)
    a_final = np.logical_or(a_aux2,a_aux).astype(int)
    
    return a_final

ef = pd.ExcelFile('SIA_ALOCACAO nov2.xlsx')

#**************************************************
#*********LEITURA DE DADOS****************
#**************************************************
#**************************************************

# Leitura do excel de entrada do LINGO
# Recorte do dominio

x_graph = []
y_graph = []
h = []

disponibilidade  = pd.read_excel(ef, 'Disponibilidade - PREENCHER', 1,usecols=['NOME', 'UF',  'DISPONIBILIDADE'])
demanda          = pd.read_excel(ef, 'Demanda - PREENCHER', 1,usecols=[ 'MISSAO', 'ATIVIDADE', 'AEROPORTO'])
arcos                = pd.read_excel(ef, 'Arcos', 1,usecols=['ORIGEM_ARCO', 'DESTINO_ARCO', 'CUSTO', 'TEMPO'])
inspecoes         = pd.read_excel(ef, 'Atividades', 1,usecols=['GRUPO_A', 'DURACAO', 'EQUIPE'])
oferta                = pd.read_excel(ef, 'Atividades', 1,usecols=['SERVIDOR',  'GRUPO_P'])

oferta               = oferta.where((oferta['GRUPO_P'].isin(demanda['ATIVIDADE'].values) == True)).dropna()
disponibilidade =  disponibilidade.where((disponibilidade['NOME'].isin(oferta['SERVIDOR'].values) == True)).dropna()
arcos                = arcos.where((arcos['ORIGEM_ARCO'].isin(disponibilidade['UF'].values) == True )).dropna()
arcos                = arcos.where((arcos['DESTINO_ARCO'].isin(demanda['AEROPORTO'].values) == True )).dropna()
inspecoes         =  inspecoes.where(inspecoes['GRUPO_A'].isin(demanda['ATIVIDADE'].values) == True).dropna()


matriz_alocado = np.zeros((len(disponibilidade),len(inspecoes)), dtype=int)

disponibilidade_aux = disponibilidade.copy()
insp_aux                  = inspecoes['GRUPO_A'].values
disp_aux                  = disponibilidade['NOME'].values
demanda_aux          = demanda['ATIVIDADE'].values

for i in range (0, len(disponibilidade)):
    for j in range (0, len(inspecoes)):
        
        if(len(oferta[oferta['SERVIDOR'].str.match(disp_aux[i]) == True].values) > 0):
            aux1 = oferta[(oferta['SERVIDOR'].str.match(disp_aux[i]) == True)]
            
            if(len(aux1[(oferta['GRUPO_P'].str.match(insp_aux[j]) == True)]) > 0):
                matriz_alocado[i][j] = 1

funcao_objetivo   =  np.zeros((len(disponibilidade),len(demanda)), dtype=int)
funcao_objetivo2 =  np.zeros((len(disponibilidade),len(demanda)), dtype=int)

#**************************************************
#*********INICIA FUNCAO OBJETIVO********
#**************************************************
#**************************************************

#inicia a função objetivo com todas as atividades possiveis para cada servidor
i = 0
for atividade in demanda['ATIVIDADE'].values:
    indice_missao        =   np.where(insp_aux == atividade)[0][0]
    funcao_objetivo[:,i] = matriz_alocado[:,indice_missao]
    i = i + 1
custo = 0

# Inicialização da funçao objetivo com as restrições devidas
# Escolhe aleatoriamente um servidor
# Pega origem e destino do servidor escolhido
# Calcula a disponibilidade do servidor e o tempo total pra realizar a missão
# Testa se o tempo pra fazer a missão cabe na disponibilidade
# Caso não caiba, sorteia outro servidor
# Sorteio de novo servidor (dica: retirar o primeiro servidor)
# Pega origem e destino
# Calcula novos tempos para possivel teste
# Atualiza a disponibilidade do servidor escolhido
# Pega origem e destino (dica: talvez não precise desse pass)
for i in range (len(demanda)):

    indice_aux  =  random.choice(np.where(funcao_objetivo[:,i] == 1)[0])
    origem        = disponibilidade.where(disponibilidade['NOME'].isin([disp_aux[indice_aux]]) == True).dropna()
    origem        = origem['UF'].values[0]
    destino       = demanda['AEROPORTO'].values[i]
    
    disponibilidade_atual = disponibilidade['DISPONIBILIDADE'].loc[disponibilidade['NOME'] == disp_aux[indice_aux]].values[0]    
    tempo_total                = inspecoes['DURACAO'].loc[inspecoes['GRUPO_A'] == demanda_aux[i]].values[0]
    tempo_total                = tempo_total + (2*(arcos['TEMPO'].loc[(arcos['ORIGEM_ARCO'] == origem) & (arcos['DESTINO_ARCO'] == destino)].values[0]))

    while(tempo_total > disponibilidade_atual):       
        indice_aux  =  random.choice(np.where(funcao_objetivo[:,i] == 1)[0])
        origem        = disponibilidade.where(disponibilidade['NOME'].isin([disp_aux[indice_aux]]) == True).dropna()
        origem        = origem['UF'].values[0]
        destino       = demanda['AEROPORTO'].values[i]
        
        disponibilidade_atual = disponibilidade['DISPONIBILIDADE'].loc[disponibilidade['NOME'] == disp_aux[indice_aux]].values[0]   
        tempo_total                = inspecoes['DURACAO'].loc[inspecoes['GRUPO_A'] == demanda_aux[i]].values[0]
        tempo_total                = tempo_total + (2*(arcos['TEMPO'].loc[(arcos['ORIGEM_ARCO'] == origem) & (arcos['DESTINO_ARCO'] == destino)].values[0]))
    
    disponibilidade['DISPONIBILIDADE'].loc[disponibilidade['NOME'] == disp_aux[indice_aux]] = disponibilidade_atual - tempo_total
    
    origem   = disponibilidade.where(disponibilidade['NOME'].isin([disp_aux[indice_aux]]) == True).dropna()
    origem   = origem['UF'].values[0]
    destino  = demanda['AEROPORTO'].values[i]
    valor      = arcos.where(arcos['ORIGEM_ARCO'].isin([origem]) == True).dropna()
    valor      = valor.where(valor['DESTINO_ARCO'].isin([destino]) == True).dropna()
    valor      = valor['CUSTO'].values[0]
    custo     = custo + valor
    
    funcao_objetivo2[indice_aux, i] = 1

#**************************************************
#*********INICIO DO ALGORITMO************
#**************************************************
#**************************************************

# a0 - original
# a1 - novo

# custos[0] - cima_desce
# custos[1] - cima_sobe
# custos[2] - baixo_sobe
# custos[3] - baixo_desce

# n = corte inicial
# k = corte iterativo e randomico
# iteracoes = quantidades de iteracoes que o algoritmo irá rodar
contador_anti_loop = 0
iteracoes = 1600
p = -1
inicio = 0
fim = len(funcao_objetivo2[:,0])
meio = int(fim/3)
a0 = funcao_objetivo2.copy()    
a1 = funcao_objetivo2.copy()    
n = 2.7
k = 0
custos = np.zeros(4, dtype=int)
c_novo = 1000000000

for x in range (0,iteracoes):
    
    #DESCE CIMA
    c_atual = calculaCusto(a0, disponibilidade, demanda, arcos)        
    a1         = salta(a0, 1,funcao_objetivo,inicio, meio)         
    a1         = mascara(a1, a0, inicio, meio)
    c_novo  = calculaCusto(a1, disponibilidade, demanda, arcos)
    k           = 0
    
    while (c_atual < c_novo):
        a1        = salta(a1, 1,funcao_objetivo,inicio, meio)
        a1        = mascara(a1, a0, inicio, meio)
        c_novo = calculaCusto(a1, disponibilidade, demanda, arcos)
        
        k+=1
        if(k > meio):
            break
        
    a_aux1       = a1.copy()
    disp_check = respeitaDisponibilidade(a1, disponibilidade, demanda, arcos,inspecoes)
    
    if(disp_check):
        custos[0] = (2 * calculaCusto(a1, disponibilidade, demanda, arcos))
    else:
        custos[0] = 1000000000

    
    #SOBE CIMA    
    a1        = salta(a0, p,funcao_objetivo,inicio, meio)
    a1        = mascara(a1, a0, inicio, meio)
    c_novo = calculaCusto(a1, disponibilidade, demanda, arcos)
    k          = 0
    
    while (c_atual < c_novo):
        a1        = salta(a1, p,funcao_objetivo,inicio, meio)
        a1        = mascara(a1, a0, inicio, meio)
        c_novo = calculaCusto(a1, disponibilidade, demanda, arcos)
        
        k+=1
        if(k > meio):
            break
        
    a_aux2       = a1.copy()
    disp_check = respeitaDisponibilidade(a1, disponibilidade, demanda, arcos,inspecoes)
    
    if(disp_check):
        custos[1] = (2 * calculaCusto(a1, disponibilidade, demanda, arcos))
    else:
        custos[1] = 1000000000

    
    #SOBE BAIXO
    a1        = salta(a0, p,funcao_objetivo,meio, fim)
    a1        = mascara(a1, a0, meio, fim)
    c_novo = calculaCusto(a1, disponibilidade, demanda, arcos)
    k           = 0
    while (c_atual < c_novo):
        a1        = salta(a1, p,funcao_objetivo,meio, fim)
        a1        = mascara(a1, a0, meio, fim)
        c_novo = calculaCusto(a1, disponibilidade, demanda, arcos)
        k+=1
        if(k > meio):
            break
        
    a_aux3       = a1.copy()
    disp_check = respeitaDisponibilidade(a1, disponibilidade, demanda, arcos,inspecoes)
    
    if(disp_check):
        custos[2] = (2 * calculaCusto(a1, disponibilidade, demanda, arcos))
    else:
        custos[2] = 1000000000

 
    #DESCE BAIXO
    a1        = salta(a0, 1,funcao_objetivo,meio, fim) 
    a1        = mascara(a1, a0, meio, fim)
    c_novo = calculaCusto(a1, disponibilidade, demanda, arcos)
    k          = 0
    
    while (c_atual < c_novo):
        a1        = salta(a1, 1,funcao_objetivo,meio, fim)
        a1        = mascara(a1, a0, meio, fim)
        c_novo = calculaCusto(a1, disponibilidade, demanda, arcos)
        
        k+=1
        if(k > meio):
            break
        
    a_aux4       = a1.copy()
    disp_check = respeitaDisponibilidade(a1, disponibilidade, demanda, arcos,inspecoes)
    
    if(disp_check):
        custos[3] = (2 * calculaCusto(a1, disponibilidade, demanda, arcos))
    else:
        custos[3] = 1000000000

    result = all(elem == custos[0] for elem in custos)

    print(x ,(2 * calculaCusto(a0, disponibilidade, demanda, arcos)), custos,  "SA")    

    menor = np.argmin(custos)

    ocorrencias_ultimo_caso = np.count_nonzero(custos == custos[menor])

    if((fim - inicio) < 2):
        n       = randint(2, 7)
        inicio = 0
        fim    = len(funcao_objetivo2[:,0])
        meio = int(fim/n)
    else:
        if((custos[menor] < 1000000000) and (result == False)):
            x_graph.append(x)
            y_graph.append(custos[menor])
            h.append(19017)
       
            if(menor == 0):
                a0    = a_aux1
                fim   = meio
                meio = int(((fim - inicio)/n)+inicio)
        
            elif(menor == 1):          
                a0    = a_aux2
                fim    = meio
                meio = int(((fim - inicio)/n)+inicio)
        
            elif(menor == 2):
                a0     = a_aux3
                inicio = meio
                meio = int(((fim - inicio)/n)+inicio)
        
            else:
                a0     = a_aux4
                inicio = meio
                meio = int(((fim - inicio)/n)+inicio)

            if(ocorrencias_ultimo_caso == 2):

                contador_anti_loop += 1                
                if(contador_anti_loop == 10):
                
                    contador_anti_loop = 0
                    n_a                         = randint(1, 4)                    
                    if(n_a == 1):
                        a0 = a_aux1
                    
                        inicio = 0
                        fim    = len(funcao_objetivo2[:,0])
                        meio = int(fim/n)
                    elif(n_a == 2):
                        a0 = a_aux2

                        inicio = 0
                        fim    = len(funcao_objetivo2[:,0])
                        meio = int(fim/n)
                    elif(n_a == 3):
                        a0 = a_aux3

                        inicio = 0
                        fim    = len(funcao_objetivo2[:,0])
                        meio = int(fim/n)
                    else:
                        a0 = a_aux4

                        inicio = 0
                        fim    = len(funcao_objetivo2[:,0])
                        meio = int(fim/n)
                
        elif((custos[menor] >= 1000000000) and (result == True) ):
            n       = randint(2, 7)
            inicio = 0
            fim    = len(funcao_objetivo2[:,0])
            meio = int(fim/n)
            
        elif((custos[menor] < 1000000000) and (result == True)):
            n       = randint(2, 7)
            inicio = 0
            fim    = len(funcao_objetivo2[:,0])
            meio = int(fim/n)            

plt.plot(x_graph, y_graph, 'r')
plt.plot(x_graph, h, 'b')
axes = plt.gca()
axes.set_ylim([0,200000])
plt.ylabel('custo')
plt.xlabel('n')
plt.show()

