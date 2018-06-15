#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 10:37:29 2018

@author: nikos
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#read the data
df = pd.read_csv('comparison.csv')

#%% plot inject vs merge
# Notes on plotting:
# https://matplotlib.org/2.0.2/examples/api/barchart_demo.html
# https://stackoverflow.com/questions/22483588/how-can-i-plot-separate-pandas-dataframes-as-subplots
plt.close('all')

#create subplots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(2*6,6))

#summarize scores per type (inject or merge)
pv = pd.pivot_table(df,values=['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4'],
                    index = ['type'])

pv.plot(kind='bar',ax=axes[0])
plt.ylabel('BLEU score')
axes[0].set_title('inject vs merge: performance')

#summarize training time per type (inject or merge)
pv = pd.pivot_table(df,values=['train_time'],
                    index = ['type'])
pv=pv/pv.values.max()#normalize time
pv.plot(kind='bar',ax=axes[1])
plt.ylabel('relative training time')
axes[1].set_title('inject vs merge: total training time')
yticks=[0,1,pv.values[1][0]]
plt.yticks(yticks,[str(np.round(x*100,1))+'%' for x in yticks])
plt.axhline(pv.values[1],linestyle='--',color='k')
plt.savefig('./figures/compare_type.png',bbox_inches='tight')

#%% plot GRU vs LSTM
plt.close('all')

#create subplots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(2*6,6))

#summarize scores per type (inject or merge)
pv = pd.pivot_table(df,values=['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4'],
                    index = ['model'])
pv.sort_index(ascending=False,inplace=True)#to show LSTM, then GRU
pv.plot(kind='bar',ax=axes[0])
plt.ylabel('BLEU score')
axes[0].set_title('LSTM vs GRU: performance')

#summarize training time per type (inject or merge)
pv = pd.pivot_table(df,values=['train_time'],
                    index = ['model'])
pv.sort_index(ascending=False,inplace=True)#to show LSTM, then GRU
pv=pv/pv.values.max()#normalize time
pv.plot(kind='bar',ax=axes[1])
plt.ylabel('relative training time')
axes[1].set_title('LSTM vs GRU: total training time')
yticks=[0,1,pv.values[1][0]]
plt.yticks(yticks,[str(np.round(x*100,1))+'%' for x in yticks])
plt.axhline(pv.values[1],linestyle='--',color='k')
plt.savefig('./figures/compare_model.png',bbox_inches='tight')

#%% for merge GRU plot dropout
plt.close('all')

#create subplots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(2*6,6))

#summarize scores per type (inject or merge)
pv = pd.pivot_table(df,values=['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4'],
                    index = ['dropout'])

pv.plot(kind='bar',ax=axes[0])
plt.ylabel('BLEU score')
axes[0].set_title('merge GRU: performance')

#summarize training time per type (inject or merge)
pv = pd.pivot_table(df,values=['train_time'],
                    index = ['dropout'])

pv=pv/pv.values.max()#normalize time
pv.plot(kind='bar',ax=axes[1])
plt.ylabel('relative training time')
axes[1].set_title('merge GRU: total training time')
yticks=[0,1,pv.values[0][0]]
plt.yticks(yticks,[str(np.round(x*100,1))+'%' for x in yticks])
plt.axhline(pv.values[0],linestyle='--',color='k')
plt.savefig('./figures/merge_GRU_dropout.png',bbox_inches='tight')





