import numpy as np
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import warnings
warnings.filterwarnings('ignore')
from contextlib import contextmanager
import mysql.connector
from sqlalchemy import create_engine
import pygsheets
from tqdm import tqdm
import yaml
import os
import binascii
import random
from multiprocessing import Process, cpu_count, Pool
from functools import partial
import numba as nb
from numba.types import List, int64

@contextmanager
def timer(name):
    """
    Taken from Konstantin Lopuhin https://www.kaggle.com/lopuhin
    in script named : Mercari Golf: 0.3875 CV in 75 LOC, 1900 s
    https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s
    """
    t0 = time.time()
    yield
    print('%s done in %.0f s'% (name,time.time()-t0))
    
def read_config_yml():
    file = open(os.getcwd() + '/Spam_detection_algo/app/config/config.yml' , "rb")
    config = yaml.load(file)
    return config

def clean_text(text):
    remove_punct_dict = dict((ord(punct), " ") for punct in string.punctuation)
    # To lower and remove punctuation
    text = text.lower().translate(remove_punct_dict)      
    # Remove extra spaces
    text = ' '.join(text.split())
    return text

# As of now, it's not possible (mostly) to jit compile a method inside a class
@nb.jit(nopython=True)
def mp_loops(y1,y2):
    i = y1[0]
    sig1 = y1[1]
    max_count = 0
    for j in range(i+1, y2.shape[0]):
        sig2 = y2[j,:]  
        if i == j:
            continue
        #count = len(set(sig1) & set(sig2)) 
        count = np.sum(sig1 == sig2)              
        if count >= max_count:
            max_count = count  
            max_ind = j
    return max_count, max_ind

class TextSim(object):
    
    def __init__(self, id_name="survey_response_id",text_column="answer",shinglesize=5):
        self.id = id_name
        self.text_column = text_column
        self.shinglesize = shinglesize
    
    def create_shingles(self, data):
        text_list = list(data[self.text_column])
        shingles_all = [self._create_shingle(text.split(" ")) for text in tqdm(text_list)]
        maxshingleID = max([max(sublist) for sublist in shingles_all])
        return shingles_all, maxshingleID
    
    def _create_shingle(self, doc):
        # 'shinglesInDoc' will hold all of the unique shingle IDs present in the current document
        shinglesInDoc = set()
        
        for index in range(0, len(doc) - (self.shinglesize - 1)):
            # Construct the shingle text by combining words together, depending on the shingle size passed      
            shingle = " ".join(doc[index:(index + self.shinglesize)])           
            # Hash the shingle to a 32-bit integer.
            shingle = bytes(shingle.strip(), encoding='utf-8')
            crc = binascii.crc32(shingle) & 0xffffffff
            # Add the hash value to the set of shingles for the current document. 
            shinglesInDoc.add(crc)         
        return shinglesInDoc
    
    def minhash_signatures(self, data, numHashes=10):        
        text_list = list(data[self.text_column])
        # We need the next largest prime number above 'maxShingleID'.
        # I looked this value up here: 
        # http://compoasso.free.fr/primelistweb/page/prime/liste_online_en.php
        nextPrime = 4294967311
        # For each of the 'numHashes' hash functions, generate a different coefficient 'a' and 'b'.   
        coeffA = self._pickRandomCoeffs(numHashes)
        coeffB = self._pickRandomCoeffs(numHashes)
        # List of documents represented as signature vectors
        signatures = []
        
        for i in tqdm(range(0, len(text_list))):

            # Convert text paragraphs into a list of words
            words = text_list[i].split(" ")   
            # Create shingles for 
            shingleIDSet = self._create_shingle(words)
            # The resulting minhash signature for this document. 
            signature = []
            
            for i in range(0, numHashes):
                # Track the lowest hash ID seen. Initialize 'minHashCode' to be greater than
                # the maximum possible value output by the hash.
                minHashCode = nextPrime + 1
                
                for shingleID in shingleIDSet:
                    # Evaluate the hash function.
                    hashCode = (coeffA[i] * shingleID + coeffB[i]) % nextPrime 
                    # Track the lowest hash code seen.
                    if hashCode < minHashCode:
                        minHashCode = hashCode
                # Add the smallest hash code value as component number 'i' of the signature.
                signature.append(minHashCode)
          
            # Store the MinHash signature for this document.
            signatures.append(signature)

        return signatures
    
    def jscore(self, len1, len2, minhash_list,numHashes=None):
        Jscores = []
        Jscore_indices = []
            
        # For each of the test documents...
        for i in tqdm(range(0, len1)):
            # Get the MinHash signature for document i.
            signature1 = minhash_list[i]     
            max_count = 0
            max_ind = 0
            # For each of the other test documents...
            for j in range(0, len2):
                # Get the MinHash signature for document j.
                signature2 = minhash_list[j]  
                if i == j:
                    continue
#                count = 0
#                # Count the number of positions in the minhash signature which are equal.
#                for k in range(0, numHashes):
#                    count = count + (signature1[k] == signature2[k])   
                count = len(set(signature1) & set(signature2))              
                if count >= max_count:
                    max_count = count
                    max_ind = j
            
            Jscores.append(max_count)   
            Jscore_indices.append(max_ind)
        return Jscores, Jscore_indices

    def mp_jscore(self, len1, len2, minhash_list,numHashes=None):              
        listA = np.array(minhash_list[:len1],dtype=np.int64)
        listA = [(i, sig) for i, sig in enumerate(listA)]
        listB = np.array(minhash_list[:len2],dtype=np.int64)
       
        max_ = len(listA)
        
        p = Pool(cpu_count())
        
        with Pool(cpu_count()) as p:
            result = list(tqdm(p.imap(partial(mp_loops, y2=listB),listA,chunksize=5), total=max_))

        Jscores = [res[0] for res in result] 
        Jscore_indices = [res[1] for res in result] 

        p.close()
        p.join()        
        return Jscores, Jscore_indices
          
    def _pickRandomCoeffs(self, k):
        # Record the maximum shingle ID that we assigned.
        maxShingleID = 2**32-1
        # Create a list of 'k' random values.
        randList = []    
        while k > 0:
            # Get a random shingle ID.
            randIndex = random.randint(0, maxShingleID)         
            # Ensure that each random number is unique.
            while randIndex in randList:
                randIndex = random.randint(0, maxShingleID)           
            # Add the random number to the list.
            randList.append(randIndex)
            k = k - 1        
        return randList


if __name__ == "__main__":
    import pickle
    with open("signatures.p","rb") as fp:
        signatures = pickle.load(fp)
                  
    # Instantiating TextSim class
    sim = TextSim(id_name = "survey_response_id",text_column = "answer",shinglesize=5)
    
    # Creating shingles from documents
    shingles, maxShingleID = sim.create_shingles(rev)
    
    # Getting minhash signatures for each document
    signatures = sim.minhash_signatures(rev,numHashes=10)
    
    # Getting Jaccard scores and document indices which produce that score
    #Jscores, Jscore_indices = sim.jscore(len(signatures), len(signatures), signatures, numHashes=10)
    Jscores, Jscore_indices = sim.mp_jscore(len(signatures), len(signatures), signatures, numHashes=10)
    print(Jscores)
    print(Jscore_indices)
        