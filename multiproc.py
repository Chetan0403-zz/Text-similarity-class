from multiprocessing import Process, cpu_count, Pool
import numpy as np
import tqdm
import pickle
import time
import pandas as pd
from functools import partial
import numba as nb
#import numba.types
from numba.types import List, int64

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed

def list_loop(y1,y2):
    Jscores = []
    Jscore_indices = []
    for i in tqdm.tqdm(range(0, (len(y1)))):
        signature1 = y1[i]     
        max_count = 0
        for j in range(0, len(y2)):
            signature2 = y2[j]  
            if i == j:
                continue
            count = len(set(signature1) & set(signature2))                
            if count >= max_count:
                max_count = count  
                max_ind = j
        Jscores.append(max_count) 
        Jscore_indices.append(max_ind) 
    return Jscores, Jscore_indices

def np_loop(y1,y2):
    Jscores = np.zeros(50,dtype=np.int64)
    Jscore_indices = np.zeros(50,dtype=np.int64)
    for i in tqdm.tqdm(range(0, y1.shape[0])):
        signature1 = y1[i,]     
        max_count = 0
        for j in range(0, y2.shape[0]):
            signature2 = y2[j,]  
            if i == j:
                continue
            count = len(set(signature1) & set(signature2))                
            if count >= max_count:
                max_count = count  
                max_ind = j
        Jscores[i,] = max_count
        Jscore_indices[i,] = max_ind
    return Jscores,Jscore_indices


@nb.jit(nopython=True)
def nb_np_loop(y1,y2):
    Jscores = np.zeros(50,dtype=np.int64)
    Jscore_indices = np.zeros(50,dtype=np.int64)
    for i in range(0, y1.shape[0]):
        signature1 = y1[i,]     
        max_count = 0
        for j in range(0, y2.shape[0]):
            signature2 = y2[j,]  
            if i == j:
                continue
            count = len(set(signature1) & set(signature2))                
            if count >= max_count:
                max_count = count  
                max_ind = j
        Jscores[i,] = max_count
        Jscore_indices[i,] = max_ind
    return Jscores,Jscore_indices


def mp_loop(y1,y2=None,r=None):  
    i, sig1 = y1
    max_count = 0
    for j,sig2 in zip(r,y2):
        if i == j:
            continue
        count = len(set(sig1) & set(sig2))                         
        if count >= max_count:
            max_count = count
            max_ind = j
    return max_count, max_ind

@nb.jit(nopython=True)
def nb_mp_loop(y1,y2,r):  
    i, sig1 = y1
    max_count = 0
    for j,sig2 in zip(r,y2):
        if i == j:
            continue
        count = len(set(sig1) & set(sig2))                         
        if count >= max_count:
            max_count = count
            max_ind = j
    return max_count, max_ind

@nb.jit(nopython=True)
def nb_list_loop(y1,y2):
    Jscores = []
    Jscore_indices = []
    for i in range(0, (len(y1))):
        signature1 = y1[i]     
        max_count = 0
        for j in range(0, len(y2)):
            signature2 = y2[j]  
            if i == j:
                continue
            count = len(set(signature1) & set(signature2))
            #count = 0               
            if count >= max_count:
                max_count = count  
                max_ind = j
        Jscores.append(max_count) 
        Jscore_indices.append(max_ind) 
    return Jscores, Jscore_indices



if __name__ == "__main__":
    
    with open("signatures.p","rb") as fp:
        signatures = pickle.load(fp)

#    #______________________________________________
#    #List loop    
#    listA = signatures[:50]
#    listB = signatures
#    
#    t0 = time.time()
#    Jscores,Jscore_indices = list_loop(listA,listB)
#    print("Elapsed time {}".format(time.time()-t0))
#    print(Jscores)
#    print(Jscore_indices)
#    
#    #______________________________________________
#    # Array loop
#    listA = signatures[:50]
#    listB = signatures
#    listA = np.array(listA,dtype=np.int64)
#    listB = np.array(listB,dtype=np.int64)
#    
#    t0 = time.time()
#    Jscores,Jscore_indices = np_loop(listA,listB)
#    print("Elapsed time {}".format(time.time()-t0))
#    print(Jscores)
#    print(Jscore_indices)
    
#    #______________________________________________
#    #Numba List loop    
#    listA = signatures[:50]
#    listB = signatures
#    
#    t0 = time.time()
#    Jscores, Jscore_indices = nb_list_loop(listA,listB)
#    print("Elapsed time {}".format(time.time()-t0))
#    print(Jscores)
#    print(Jscore_indices)
    
#    #______________________________________________
#    # Numba Array loop
#    listA = signatures[:50]
#    listB = signatures
#    listA = np.array(listA,dtype=np.int64)
#    listB = np.array(listB,dtype=np.int64)
#    
#    t0 = time.time()
#    Jscores,Jscore_indices = nb_np_loop(listA,listB)
#    print("Elapsed time {}".format(time.time()-t0))
#    print(Jscores)
#    print(Jscore_indices)
#    
    #______________________________________________
    # Multiproc loop
    listA = signatures[:100]
    listB = signatures    
    listA = [(i, sig) for i, sig in enumerate(signatures[:100])]
    listB = signatures
    rng = [i for i in range(0,len(listB))]
   
    p = Pool(cpu_count())
    
    t0 = time.time()

    # Trying tqdm with multiprocessing
    max_ = len(listA)
   
    with Pool(cpu_count()) as p:
        result = list(tqdm.tqdm(p.imap(partial(mp_loop, y2=listB, r=rng),listA,chunksize=10), total=max_))

    #result = p.map(partial(loop, y2=listB, r=rng),listA,chunksize=20)
    Jscores = [res[0] for res in result] 
    Jscore_indices = [res[1] for res in result] 
    print("Elapsed time {}".format(time.time()-t0))
    print(Jscores)
    print(Jscore_indices)

    p.close()
    p.join()
#
#    #______________________________________________
#    # Multiproc numba loop
#    listA = signatures[:50]
#    listB = signatures    
#    listA = [(i, sig) for i, sig in enumerate(signatures[:50])]
#    listB = signatures
#    rng = [i for i in range(0,len(listB))]
#   
#    p = Pool(cpu_count())
#    
#    t0 = time.time()
#
#    # Trying tqdm with multiprocessing
#    max_ = len(listA)
#   
#    with Pool(cpu_count()) as p:
#        result = list(tqdm.tqdm(p.imap(partial(nb_mp_loop, y2=listB, r=rng),listA,chunksize=10), total=max_))
#
#    #result = p.map(partial(loop, y2=listB, r=rng),listA,chunksize=20)
#    Jscores = [res[0] for res in result] 
#    Jscore_indices = [res[1] for res in result] 
#    print("Elapsed time {}".format(time.time()-t0))
#    print(Jscores)
#    print(Jscore_indices)
#
#    p.close()
#    p.join()


"""
Run-time comparisons:
    
Samples = 50

List double loop: 42s
List double loop, nopython=False: 37s
Array double loop: 1m 48s
Numba, Array double loop: 24.7s
Multiproc, List double loop: 32s
"""