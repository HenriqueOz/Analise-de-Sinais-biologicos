# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 20:42:33 2020

@author: Henri
"""
import numpy as np
import time
start_time = time.time()
n_points = 100000
# Criar vetor para armazenar o dado aquisitado
signal = np.zeros(n_points)
import numpy as np
import time
start_time = time.time()
# Aquisitar o dado em si
signal = np.random.rand(n_points)

teste = np.std(signal)

print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
n_points = 100000
# Criar vetor para armazenar o dado aquisitado
signal = np.zeros(n_points)
# Aquisitar o dado em si
signal = np.random.rand(n_points)

teste = np.histogram(a = signal, bins = 1000)

print("--- %s seconds ---" % (time.time() - start_time))