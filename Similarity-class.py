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
from numba import jit

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
    
    def jscore(self, len1, len2, minhash_list,numHashes=10):
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
        
    with timer("Fetching data from feedback prod db"):
        
        # Establishing connection with feedback prod db
        configuration = read_config_yml()
        conn = mysql.connector.connect(user = configuration['mysql']['user'],
                               password = configuration['mysql']['password'],
                               host = configuration['mysql']['host'],
                               database = configuration['mysql']['database'], 
                               port = configuration['mysql']['port'])
        
        mycursor = conn.cursor()
        print("Connection to feedback db established")
        
        # Columns with which dataframe needs to be saved
        columns = ['doctor_id', 'survey_response_id', 'recommendation', 'created_at', 
                   'rm_deleted_at', 'sr_deleted_at', 'sra_deleted_at', 's_deleted_at',
                   'is_spam','user_verified','is_contested','mobile', 'channel', 
                   'answer', 'status', 'owning_service', 'anonymous']
        
        # Fetch query
        mycursor.execute("""select doctor_id, 
                            rm.survey_response_id as survey_response_id,
                            rm.recommendation, 
                            rm.created_at, 
                            rm.deleted_at as rm_deleted_at,
                            sr.deleted_at as sr_deleted_at,
                            sra.deleted_at as sra_deleted_at,
                            s.deleted_at as s_deleted_at,
                            s.is_spam,
                            s.user_verified,
                            s.is_contested,
                            r.mobile, 
                            channel, 
                            answer, 
                            rm.status, 
                            c.owning_service, 
                            s.anonymous                          
                            FROM 
                            feedback.survey_responses AS sr
                            JOIN feedback.survey_response_answers AS sra ON sr.id=sra.survey_response_id
                            JOIN feedback.review_moderations AS rm ON rm.survey_response_id=sra.survey_response_id
                            JOIN feedback.surveys s ON sr.survey_id = s.id
                            JOIN feedback.campaigns c ON s.campaign_id = c.id
                            JOIN feedback.respondees as r on s.respondee_id=r.id
                            WHERE rm.review_for = 'DOCTOR'""")
        
        # Store fetched data into dataframe
        df_rev_text = mycursor.fetchall()
        df_rev_text = pd.DataFrame(df_rev_text, columns = columns)
       
        # Closing feedback prod db connection
        conn.close()    
        mycursor.close()
 
    with timer("Computing review similarity scores"):
        """ 
        1. Computation of review text similarity below.
        2. valid incremental reviews for each doctor are compared with each other and with past reviews on a range of 1-10 ngrams for similarity
        3. Cosine similarity computed
        """      
        # Shortlisting reviews for similarity checking
        rev = df_rev_text[(df_rev_text['rm_deleted_at'].isnull()) &
                          (df_rev_text['sra_deleted_at'].isnull()) &
                          (df_rev_text['sr_deleted_at'].isnull()) &
                          (df_rev_text['s_deleted_at'].isnull()) &
                          (df_rev_text['is_spam'] == 0) &
                          (df_rev_text['user_verified'] == 1) &
                          (df_rev_text['is_contested'] == 0) &
                          (df_rev_text['status'] == 'PUBLISHED')]
        
        import pickle
        with open("df_rev_text.p","rb") as fp:
            df_rev_text = pickle.load(fp)   
        
        rev = df_rev_text[~df_rev_text['answer'].isnull()]
        
        # Word count. Drop reviews with word count less than shingle size requirements
        rev['answer'] = rev['answer'].apply(lambda x: clean_text(x))   
        rev['word_count'] = rev['answer'].apply(lambda text: sum(1 for word in text.split()))
        rev = rev[rev['word_count'] >= 20]

        # List of survey ids
        sur_ids = list(rev['survey_response_id'])
                
        # Instantiating TextSim class
        sim = TextSim(id_name = "survey_response_id",text_column = "answer",shinglesize=5)
        
        # Creating shingles from documents
        shingles, maxShingleID = sim.create_shingles(rev)
        
        # Getting minhash signatures for each document
        signatures = sim.minhash_signatures(rev,numHashes=10)
        
        # Getting Jaccard scores and document indices which produce that score
        Jscores, Jscore_indices = sim.jscore(100, len(signatures), signatures, numHashes=10)
        
        # Debugging. Score 5
        print("{}\n\n{}".format(rev.iloc[50,:]['answer'],
                              rev.iloc[374,:]['answer']))