# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 22:07:09 2020

@author: Henrique Oliveira dos Santos
Aluno especial no mestrado de engenharia elétrica da UFSCar
Matéria: Análise de sinais biológicos
Aula: Semana 02
Exercício: Capítulo 8 - Código 3 - Rect2pol e Pol2rect
RECTANGULAR-TO-POLAR & POLAR-TO-RECTANGULAR CONVERSION
"""
import numpy as np
import time
import matplotlib.pyplot as plt

def polrect(MagX, PhaseX):
    M = MagX.shape[0]
    ReX = np.zeros(M)
    ImX = np.zeros(M)

    for k in range(0,M-1):
        ReX[k] = MagX[k] * np.cos(PhaseX[k])
        ImX[k] = MagX[k] * np.sin(PhaseX[k])
    return ReX, ImX

def rectpol(ReX, ImX):
    M = ReX.shape[0]
    MagX = np.zeros(M)
    PhaseX = np.zeros(M)

    for k in range(0,M-1):
        MagX[k] = np.sqrt(ReX[k]**2 + ImX[k]**2)
        if (ReX[k] == 0):
            ReX[k] = 1E-20
        PhaseX[k] = np.arctan(ImX[k] / ReX[k])
        if (ReX[k] < 0):
            if (ImX[k] < 0):
                PhaseX[k] -= np.pi
            else:
                PhaseX[k] += np.pi

    return MagX, PhaseX

# %% Teste das funções

start_time = time.time()
a,b = rectpol(REX,IMX)
print("Com %s amostras" % N)

print("Retangular para polar pelo livro: %s segundos" % (time.time() - start_time))
c,d = polrect(a,b)

start_time = time.time()
e = np.angle(REIMX)
f = np.abs(REIMX)

print("Retangular pela nativa: %s segundos" % (time.time() - start_time))



plt.figure()
plt.title("Parte real")
plt.plot(REX, marker ='o',label ='Matriz real antes')
plt.plot(c, label = 'Matriz real após')
plt.legend()
plt.figure()
plt.title("Parte imaginária")
plt.plot(IMX, marker ='o',label ='Matriz imag antes')
plt.plot(d, label = 'Matriz imag após')
plt.legend()