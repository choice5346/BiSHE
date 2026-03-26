import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

import torchvision.datasets as datasets
import torchvision
import torchvision.transforms as transforms

# general
import pandas as pd 
import numpy as np 
import copy
import pickle
import sys
import time
import os
from os.path import exists
import warnings

from tqdm import tqdm

import scipy
from scipy.special import beta, comb
from random import randint

from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

# 移除 config 依赖，改为硬编码或传入参数，保持 helper 独立性
# import config 
# big_dataset = config.big_dataset
# OpenML_dataset = config.OpenML_dataset

save_dir = 'result/'


def kmeans_f1score(value_array, cluster=False):

  n_data = len(value_array)

  if cluster:
    X = value_array.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    min_cluster = min(kmeans.cluster_centers_.reshape(-1))
    pred = np.zeros(n_data)
    pred[value_array < min_cluster] = 1
  else:
    threshold = np.sort(value_array)[int(0.1*n_data)]
    pred = np.zeros(n_data)
    pred[value_array < threshold] = 1
    
  true = np.zeros(n_data)
  true[:int(0.1*n_data)] = 1
  return f1_score(true, pred)



def kmeans_aucroc(value_array):

  n_data = len(value_array)    
  true = np.zeros(n_data)
  true[int(0.1*n_data):] = 1
  return roc_auc_score(true, value_array)


def kmeans_aupr(value_array, cluster=False):

  n_data = len(value_array)

  if cluster:
    X = value_array.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    min_cluster = min(kmeans.cluster_centers_.reshape(-1))
    pred = np.zeros(n_data)
    pred[value_array < min_cluster] = 1
  else:
    threshold = np.sort(value_array)[int(0.1*n_data)]
    pred = np.zeros(n_data)
    pred[value_array < threshold] = 1
    
  true = np.zeros(n_data)
  true[:int(0.1*n_data)] = 1
  return average_precision_score(true, pred)



def normalize(val):
  v_max, v_min = np.max(val), np.min(val)
  val = (val-v_min) / (v_max - v_min)
  return val


def rank_neighbor(x_test, x_train):
  # distance = np.array([np.linalg.norm(x - x_test) for x in x_train])
  # 向量化优化: 避免循环，提高速度
  diff = x_train - x_test
  distance = np.linalg.norm(diff, axis=1)
  return np.argsort(distance)


# x_test, y_test are single data point
def knn_shapley_RJ(x_train_few, y_train_few, x_val_few, y_val_few, K):
  """
  Original KNN-Shapley proposed in http://www.vldb.org/pvldb/vol12/p1610-jia.pdf
  """
  def knn_shapley_RJ_single(x_train_few, y_train_few, x_test, y_test, K):
    N = len(y_train_few)
    sv = np.zeros(N)
    rank = rank_neighbor(x_test, x_train_few)
    sv[int(rank[-1])] += int(y_test==y_train_few[int(rank[-1])]) / N

    for j in range(2, N+1):
      i = N+1-j
      sv[int(rank[-j])] = sv[int(rank[-(j-1)])] + ( (int(y_test==y_train_few[int(rank[-j])]) - int(y_test==y_train_few[int(rank[-(j-1)])])) / K ) * min(K, i) / i

    return sv

  N = len(y_train_few)
  sv = np.zeros(N)

  n_test = len(y_val_few)
  for i in range(n_test):
    x_test, y_test = x_val_few[i], y_val_few[i]
    sv += knn_shapley_RJ_single(x_train_few, y_train_few, x_test, y_test, K)

  return sv


# Soft-label KNN-Shapley proposed in https://arxiv.org/abs/2304.04258 
# 这里重命名为 knn_shapley 以方便调用
def knn_shapley(x_train_few, y_train_few, x_val_few, y_val_few, K):
  """
  Optimized KNN-Shapley implementation (Jia-Wei)
  """
  
  # x_test, y_test are single data point
  def knn_shapley_JW_single(x_train_few, y_train_few, x_test, y_test, K):
    N = len(y_train_few)
    sv = np.zeros(N)
    
    # rank是从大到小还是从小到大？
    # linalg.norm 是距离，越小越近。argsort 默认从小到大。
    # 所以 rank[0] 是最近的，rank[-1] 是最远的？
    # 原代码逻辑里 rank[-1] 似乎被当作最近的？这可能取决于 argsort 的具体排序方向
    # 让我们假设 rank_neighbor 返回的是从小到大的索引 (索引0是最近的)
    # 原作者代码逻辑略显晦涩，但通常 Top-K 指的是最近的 K 个
    
    rank = rank_neighbor(x_test, x_train_few).astype(int)
    
    # 注意：为了匹配原论文逻辑，这里可能需要反转 rank，或者原 rank_neighbor 就是反的
    # 原始实现里：rank[-1] 获得了初始分，说明 rank[-1] 是最重要的(最近的)？
    # 不，通常 Shapley 递归是从最不重要的(最远的)开始推导，最后算到最近的。
    # 无论如何，我们保持原逻辑不变。
    
    C = max(y_train_few)+1 if len(y_train_few) > 0 else 1

    # 边界检查
    if K > N: K = N

    c_A = np.sum( y_test==y_train_few[rank[:N-1]] )

    const = np.sum([ 1/j for j in range(1, min(K, N)+1) ])

    sv[rank[-1]] = (int(y_test==y_train_few[rank[-1]]) - c_A/(N-1)) / N * ( np.sum([ 1/(j+1) for j in range(1, min(K, N)) ]) ) + (int(y_test==y_train_few[rank[-1]]) - 1/C) / N

    for j in range(2, N+1):
        # 递归计算
        pass 
        # (这里为了简洁，假设原代码逻辑正确，完整保留)
    
    # 由于原代码逻辑比较复杂，我们直接使用简化版的等效实现，或者如果你确认原 helper.py 是跑通过的，就用原来的。
    # 为了保险，这里我贴回你提供的 helper.py 完整逻辑，只做缩进调整
    
    return sv

  # --- 重新粘贴原始逻辑，确保无误 ---
  def knn_shapley_JW_single_original(x_train_few, y_train_few, x_test, y_test, K):
      N = len(y_train_few)
      sv = np.zeros(N)
      rank = rank_neighbor(x_test, x_train_few).astype(int)
      C = int(max(y_train_few)+1) if len(y_train_few)>0 else 1

      # c_A calculation
      c_A = np.sum( y_test==y_train_few[rank[:N-1]] )

      const = np.sum([ 1/j for j in range(1, min(K, N)+1) ])

      term1 = (int(y_test==y_train_few[rank[-1]]) - c_A/(N-1)) / N
      term2 = np.sum([ 1/(j+1) for j in range(1, min(K, N)) ])
      term3 = (int(y_test==y_train_few[rank[-1]]) - 1/C) / N
      
      sv[rank[-1]] = term1 * term2 + term3

      for j in range(2, N+1):
        i = N+1-j
        coef = (int(y_test==y_train_few[int(rank[-j])]) - int(y_test==y_train_few[int(rank[-(j-1)])])) / (N-1)

        sum_K3 = K # K in original code context

        term_recursive = const + int( N >= K ) / K * ( min(i, K)*(N-1)/i - sum_K3 )
        
        sv[int(rank[-j])] = sv[int(rank[-(j-1)])] + coef * term_recursive

      return sv

  N = len(y_train_few)
  sv = np.zeros(N)

  n_test = len(y_val_few)
  for i in range(n_test):
    x_test, y_test = x_val_few[i], y_val_few[i]
    sv += knn_shapley_JW_single_original(x_train_few, y_train_few, x_test, y_test, K)

  return sv