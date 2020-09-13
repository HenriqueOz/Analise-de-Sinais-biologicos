# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 20:04:33 2020

@author: Henri
"""

import numpy as np
import time
start_time = time.time()

n_pontos = 100000   

signal = np.zeros(n_pontos)
signal = np.random.random(size = n_pontos)*10

histogram = np.zeros(1000)

for data_point in signal:
    sinal_ceifado = int(data_point*100)
    histogram[sinal_ceifado] = histogram[sinal_ceifado] + 1


print("--- %s seconds ---" % (time.time() - start_time))

teste, bins = np.histogram(a = signal, bins = 1000)
    