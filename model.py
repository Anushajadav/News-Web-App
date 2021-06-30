from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import pandas as pd



news=pd.read_csv("uci-news-aggregator.csv")



news.head()



news.dtypes

 


news['CATEGORY'].astype(str)

 


news['CATEGORY']=news['CATEGORY'].astype(str)

 


news.dtypes

 


import pandas as pd

 


news.CATEGORY=news.CATEGORY.astype(str)

 


news.dtypes

 

import re
import string
def normalize_text(s):
    s = s.lower()
    s = re.sub('(https?:\/\/)(\s)?(www\.)?(\s?)(\w+\.)*([\w\-\s]+\/)*([\w-]+)\/?',' ',s)
    s = re.sub('\s\W',' ',s)
    s = re.sub('\W\s',' ',s)
    s = re.sub("[0-9]+", " ",s)
    s = re.sub(r"\b[a-z]\b", " ", s)
    for ch in string.punctuation:                                                                                                     
        s = s.replace(ch, " ")
    s = re.sub('\s+',' ',s)
    s = s.strip()
    return s

news['TITLE'] = [normalize_text(s) for s in news['TITLE']]

 

news.head()

 


lens = [len(s) for s in news['TITLE']]

 
 

import numpy as np

 

np.max(lens)


 



from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder() #to convert non-numeric values to numeric values
x = news['TITLE']
y = encoder.fit_transform(news['CATEGORY'])


print(y)



print(x) 



 

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
count_vect = CountVectorizer(stop_words='english')



X_counts = count_vect.fit_transform(x)

print("the size of matrix is",X_counts.shape[0],"X",X_counts.shape[1])


import pickle
pickle.dump(count_vect.vocabulary_, open("count_vector.pkl","wb"))



tfidf_transformer =TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)

pickle.dump(tfidf_transformer, open("tfidf.pkl","wb"))

nb_model= MultinomialNB()



nb_model.fit(X_tfidf,y)




 


pickle.dump(nb_model, open("nb_model.pkl", "wb"))
