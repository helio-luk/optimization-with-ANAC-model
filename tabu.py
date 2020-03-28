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




def isInList(a0, tabuList):
    r = False
    for i in tabuList:
        if(np.array_equal(i, a0) == True):
            r = True
            break
        else:
            r = False
    return r

def getNbhd (a0,funcao_objetivo,passo):
    x = len(a0[0,:])
    aList = []
    p = -1
    a1 = a0.copy()
    for i in range (0, int(x/passo)):
        fator = random.randint(1, 5)
        for j in range (0,fator):
            a0 = salta(a0,1,funcao_objetivo,(passo*i), passo)
            a1 = salta(a1,1,funcao_objetivo,(passo*i), passo)
        aList.append(salta(a0,1,funcao_objetivo,(passo*i), passo))
        aList.append(salta(a1,p,funcao_objetivo,(passo*i), passo))
    return aList


def salta(a0, direcao,FO,posicao_passo,tamanho_passo):
    a_aux = a0.copy()
    
    if(tamanho_passo+posicao_passo < len(FO[0,:] )):
        final = tamanho_passo+posicao_passo
    else:
        final = len(FO[0,:] )

      
    for i in range (posicao_passo, final):
        
        #fazer o recorte do salto em cima dos servidores_aptos tambem
        servidores_aptos = np.where(FO[:,i] == 1)[0]        
            
        try:            
            indice_atual = np.where(a0[:,i] == 1)[0][0]                                          
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



#**************************************************
#*********INICIA TABU**************************
#**************************************************
#**************************************************
s0 = geraIndividuo(funcao_objetivo, disponibilidade,disp_aux,inspecoes, demanda_aux,arcos)
sBest = s0
bestCandidate = s0
tabuList = []
tabuList.append(s0)

#não tá funcionadno

for i in range (0,1600):
    passoAleatorio = random.randint(1, 7)
    sNbhd = getNbhd(bestCandidate,funcao_objetivo,passoAleatorio)
    #for j in sNbhd:
        #print(respeitaDisponibilidade(j, disponibilidade, demanda, arcos,inspecoes))
            
    bestCandidate = sNbhd[0]
    
    for sCandidate in sNbhd:
        
        custoCandidate = calculaCusto(sCandidate, disponibilidade, demanda, arcos)
        custoBest          = calculaCusto(bestCandidate, disponibilidade, demanda, arcos)

        #não tá na lista?
        #custo do canditado é menor que o custo do best?
        #candidato respeita disponibilidade?
        if((isInList(sCandidate, tabuList) == False) and (custoCandidate < custoBest) and (respeitaDisponibilidade(sCandidate, disponibilidade, demanda, arcos,inspecoes) == True) and (valida(sCandidate) == True)):
            bestCandidate = sCandidate
            
    custoBest = calculaCusto(bestCandidate, disponibilidade, demanda, arcos)    
    custosBest = calculaCusto(sBest, disponibilidade, demanda, arcos)
    tabuList.append(bestCandidate)

    x_graph.append(i)
    y_graph.append(custosBest)
    h.append(19017)
    
    print(i,int(custosBest), int(custoBest),"TABU")
    if(custoBest < custosBest):
        sBest = bestCandidate

    if(len(tabuList) > 60):
        del tabuList[0]
        
plt.plot(x_graph, y_graph, 'r')
plt.plot(x_graph, h, 'b')
axes = plt.gca()
axes.set_ylim([0,200000])
plt.ylabel('custo')
plt.xlabel('iterações')
plt.show()

