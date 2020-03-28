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

def valida(a0):
    resultado = True
    for i in range (0, len(a0[0,:])):
        indice_aux = np.where(a0[:,i] == 1)
        if(len(indice_aux[0]) != 1):
            resultado = False
    return resultado


# pega metade de cada pai
# caso apareca mais de uma pessoa por missao, escolhe aleatoriamente
def crossover(a0, a1,funcao_objetivo): 
    #b = np.logical_or(a0,a1).astype(int)
    b =  mascara(a0, a1, 0, 20)
    for i in range (0, len(b[0,:])):
        indice_aux = np.where(b[:,i] == 1)
        if(len(indice_aux[0]) == 2):
            indice_random  =  random.choice(indice_aux[0])
            b[indice_random,i] = 0
        elif(len(indice_aux[0]) == 0):
            indice_aux = np.where(funcao_objetivo[:,i] == 1)
            indice_random  =  random.choice(indice_aux[0])
            b[indice_random,i] = 1
    return b


def geraIndividuo(funcao_objetivo, disponibilidade,disp_aux,inspecoes, demanda_aux,arcos):
    funcao_objetivo2 =  np.zeros((len(disponibilidade),len(demanda_aux)), dtype=int)
    custo = 0
    disponibilidade_copia = disponibilidade.copy()
    for i in range (len(demanda)):

        indice_aux  =  random.choice(np.where(funcao_objetivo[:,i] == 1)[0])
        origem        = disponibilidade_copia.where(disponibilidade_copia['NOME'].isin([disp_aux[indice_aux]]) == True).dropna()
        origem        = origem['UF'].values[0]
        destino       = demanda['AEROPORTO'].values[i]
    
        disponibilidade_atual = disponibilidade_copia['DISPONIBILIDADE'].loc[disponibilidade_copia['NOME'] == disp_aux[indice_aux]].values[0]    
        tempo_total                = inspecoes['DURACAO'].loc[inspecoes['GRUPO_A'] == demanda_aux[i]].values[0]
        tempo_total                = tempo_total + (2*(arcos['TEMPO'].loc[(arcos['ORIGEM_ARCO'] == origem) & (arcos['DESTINO_ARCO'] == destino)].values[0]))

        while(tempo_total > disponibilidade_atual):       
            indice_aux  =  random.choice(np.where(funcao_objetivo[:,i] == 1)[0])
            origem        = disponibilidade_copia.where(disponibilidade['NOME'].isin([disp_aux[indice_aux]]) == True).dropna()
            origem        = origem['UF'].values[0]
            destino       = demanda['AEROPORTO'].values[i]
        
            disponibilidade_atual = disponibilidade_copia['DISPONIBILIDADE'].loc[disponibilidade_copia['NOME'] == disp_aux[indice_aux]].values[0]   
            tempo_total                = inspecoes['DURACAO'].loc[inspecoes['GRUPO_A'] == demanda_aux[i]].values[0]
            tempo_total                = tempo_total + (2*(arcos['TEMPO'].loc[(arcos['ORIGEM_ARCO'] == origem) & (arcos['DESTINO_ARCO'] == destino)].values[0]))
    
        disponibilidade_copia['DISPONIBILIDADE'].loc[disponibilidade_copia['NOME'] == disp_aux[indice_aux]] = disponibilidade_atual - tempo_total # aqui tá dando rpoblema, eu preciso fazer uma copia
    
        origem   = disponibilidade_copia.where(disponibilidade_copia['NOME'].isin([disp_aux[indice_aux]]) == True).dropna()
        origem   = origem['UF'].values[0]
        destino  = demanda['AEROPORTO'].values[i]
        valor      = arcos.where(arcos['ORIGEM_ARCO'].isin([origem]) == True).dropna()
        valor      = valor.where(valor['DESTINO_ARCO'].isin([destino]) == True).dropna()
        valor      = valor['CUSTO'].values[0]
        custo     = custo + valor
    
        funcao_objetivo2[indice_aux, i] = 1


    return funcao_objetivo2.copy()

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
    return custo_total * 2
    

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
#x_graph.append(0)
y_graph = []
#y_graph.append(0)
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
a = []
a.append(geraIndividuo(funcao_objetivo, disponibilidade,disp_aux,inspecoes, demanda_aux,arcos))
a.append(geraIndividuo(funcao_objetivo, disponibilidade,disp_aux,inspecoes, demanda_aux,arcos))
a.append(geraIndividuo(funcao_objetivo, disponibilidade,disp_aux,inspecoes, demanda_aux,arcos))
a.append(geraIndividuo(funcao_objetivo, disponibilidade,disp_aux,inspecoes, demanda_aux,arcos))

custos_pais = np.zeros(4, dtype=int)
for i in range (0,4):
    custos_pais[i] = calculaCusto(a[i], disponibilidade, demanda, arcos)

#**************************************************
#*********INICIA GENETIC*********************
#**************************************************
#**************************************************
for j in range (0,1600):
    # 6 cruzamentos possiveis
    b = []
    b.append(crossover(a[0], a[1],funcao_objetivo))
    b.append(crossover(a[1], a[2],funcao_objetivo))
    b.append(crossover(a[2], a[3],funcao_objetivo))
    b.append(crossover(a[1], a[3],funcao_objetivo))
    b.append(crossover(a[0], a[2],funcao_objetivo))
    b.append(crossover(a[0], a[3],funcao_objetivo))

    #Calcula custo apenas dos validos
    custos_filhos = np.zeros(6, dtype=int)
    for i in range (0,6):
        v = valida(b[i])
        d = respeitaDisponibilidade(b[i], disponibilidade, demanda, arcos,inspecoes)
        if(v == True and d == True):
            custos_filhos[i] =  calculaCusto(b[i], disponibilidade, demanda, arcos)
        else:
            custos_filhos[i] = 1000000000
    #pega o menor valor dos filhos e insere nos pais
    quantidades_validos  = np.where(custos_filhos < 1000000000)[0]
    if (len(quantidades_validos) > 0):        
        for i in range (0,len(quantidades_validos)):
            if(custos_pais[np.argmax(custos_pais)] >  custos_filhos[quantidades_validos[i]]):
                a[np.argmax(custos_pais)] = b[quantidades_validos[i]]
                
    for i in range (0,4):
        custos_pais[i] = calculaCusto(a[i], disponibilidade, demanda, arcos)
    # a cada 5 iterações muta os pais
    if ((j % 7) == 0):
        #k = randint(0, 3)
        a[np.argmin(custos_pais) - 1] = geraIndividuo(funcao_objetivo, disponibilidade,disp_aux,inspecoes, demanda_aux,arcos)
        a[np.argmin(custos_pais) - 2] = geraIndividuo(funcao_objetivo, disponibilidade,disp_aux,inspecoes, demanda_aux,arcos)
        a[np.argmin(custos_pais) - 3] = geraIndividuo(funcao_objetivo, disponibilidade,disp_aux,inspecoes, demanda_aux,arcos)
        
    menor = np.argmin(custos_pais)
    print(j, custos_pais[menor], custos_pais,"GA" )
    
    x_graph.append(j)
    y_graph.append(custos_pais[menor])
    h.append(19017)
    
plt.plot(x_graph, y_graph, 'r')
plt.plot(x_graph, h, 'b')
axes = plt.gca()
axes.set_ylim([0,200000])
plt.ylabel('custo')
plt.xlabel('iterações')
plt.show()

