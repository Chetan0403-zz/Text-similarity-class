from multiprocessing import Process, cpu_count, Pool
import numpy as np
import tqdm
import pickle
import time
import pandas as pd
from functools import partial

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

#@timeit    
#def loop(y1,y2):
#    Jscores = []
#    Jscore_indices = []
#    for i in range(0, len(y1)):
#        signature1 = y1[i]     
#        max_count = 0
#        for j in range(0, len(y2)):
#            signature2 = y2[j]  
#            if i == j:
#                continue
#            count = len(set(signature1) & set(signature2))                
#            if count >= max_count:
#                max_count = count  
#                max_ind = j
#        Jscores.append(max_count) 
#        Jscore_indices.append(max_ind) 
#    return Jscores, Jscore_indices

def loop(y1,y2=None,r=None):  
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

if __name__ == "__main__":
    
    with open("signatures.p","rb") as fp:
        signatures = pickle.load(fp)

#    # Normal loop    
#    listA = signatures[:45]
#    listB = signatures
#    Jscores,Jscore_indices = loop(listA,listB)
   
    listA = [(i, sig) for i, sig in enumerate(signatures[:45])]
    listB = signatures
    rng = [i for i in range(0,len(listB))]
   
    p = Pool(cpu_count())
    
    t0 = time.time()
    result = p.map(partial(loop, y2=listB, r=rng),listA)
    Jscores = [res[0] for res in result] 
    Jscore_indices = [res[1] for res in result] 
    print("Elapsed time {}".format(time.time()-t0))
    print(Jscores)
    print(Jscore_indices)

    p.close()
    p.join()
   
#    # Trying tqdm with multiprocessing
#    max_ = len(listA)
    
#    with tqdm.tqdm(total=max_) as pbar:
#        for result in tqdm.tqdm(p.imap_unordered(partial(loop, y2=listB, r=rng),listA)):
#            pbar.update()
    
#    with Pool(cpu_count()) as p:
#        result = list(tqdm.tqdm(p.imap(partial(loop, y2=listB, r=rng),listA), total=max_))
    




