# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 12:17:44 2019

@author: Lenovo
"""

import numpy as np
import pandas as pd

import seaborn as sns #for plotting
import matplotlib.pyplot as plt
from matplotlib import pylab
from scipy import stats

from warnings import filterwarnings
filterwarnings('ignore')

#data = pd.read_csv('dfMedidas_complete.csv',delimiter=None)
#data=pd.read_csv('dfMedidas1.csv')
data=pd.read_csv('dfMedidas_completes.csv', sep=';')
df = data.copy()
df.head()
df.shape