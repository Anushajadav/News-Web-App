import collections
from flask import request
from pymongo import message
import requests
import numpy as np
import pymongo
 
from sklearn import metrics
from sklearn.metrics import confusion_matrix 
from flask import Flask,jsonify,render_template
from bson.json_util import dumps
import pickle
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import dns
import json
import pandas as pd
from pymongo import MongoClient

client = pymongo.MongoClient("mongodb+srv://anusha_123:anusha_12345@cluster0.bcowz.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
db = client['news']
db2=client['contactUs']
 
app = Flask(__name__)
 
 
@app.route('/')
def home():

    entertainment=requests.get('https://newsapi.org/v2/top-headlines?country=in&category=entertainment&apiKey=eafd81ede8dd42ffa61e0d0adf6a869c')
    entertainment=entertainment.json()
    entertainment=entertainment["articles"]
    entertainment=pd.DataFrame(entertainment)
    entertainment["category"]=1
 

    health=requests.get('https://newsapi.org/v2/top-headlines?country=in&category=health&apiKey=eafd81ede8dd42ffa61e0d0adf6a869c')
    health=health.json()
    health=health["articles"]
    health=pd.DataFrame(health)
    health["category"]=2

    science=requests.get('https://newsapi.org/v2/top-headlines?country=in&category=science&apiKey=eafd81ede8dd42ffa61e0d0adf6a869c')
    science=science.json()
    science=science["articles"]
    science=pd.DataFrame(science)
    science["category"]=3

    technology=requests.get('https://newsapi.org/v2/top-headlines?country=in&category=technology&apiKey=eafd81ede8dd42ffa61e0d0adf6a869c')
    technology=technology.json()
    technology=technology["articles"]
    technology=pd.DataFrame(technology)
    technology["category"]=3
 
    business=requests.get('https://newsapi.org/v2/top-headlines?country=in&category=business&apiKey=eafd81ede8dd42ffa61e0d0adf6a869c')
    business=business.json()
    business=business["articles"]
    business=pd.DataFrame(business)
    business["category"]=0
    data_test=pd.concat([business,technology,science,health,entertainment],ignore_index=True)
    data_test =data_test.sample(frac=1).reset_index(drop=True)
    print(data_test.shape)
    data_test.head()
    headline=data_test.to_dict("records")
    collection5=db["headline"]
    collection5.drop()
    collection5=db["headline"]
    collection5.insert_many(headline)
    cursor=db.headline.find()
    list_cur = list(cursor)
    json_data= dumps(list_cur) 
    data=json.loads(json_data)
    news=[]
    for i in range(len(data)):
        news.append(data[i]['title'])
    mylist = zip(news)

 
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
    data_test['title']= [normalize_text(s) for s in data_test['title']]
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    import pickle
    loaded_vec = CountVectorizer(vocabulary=pickle.load(open("count_vector.pkl", "rb")))
    loaded_tfidf = pickle.load(open("tfidf.pkl","rb"))
    loaded_model = pickle.load(open("nb_model.pkl","rb"))

    X_counts_test = loaded_vec.transform(data_test['title'])
    print("the size of matrix is",X_counts_test.shape[0],"X",X_counts_test.shape[1])
    X_tfidf_test= loaded_tfidf.transform(X_counts_test)
    y_pred=loaded_model.predict(X_tfidf_test)
    
    y_test=data_test["category"]
    y_test=y_test.values

 
    print("Accuracy of prediction: ",metrics.accuracy_score(y_test, y_pred)*100,"%")
    #uncomment when it is required to predict.

    # cm=confusion_matrix(y_test, y_pred)
    # print("Confusion matrix:")
    # print(cm)
    # FP=cm.sum(axis=0)-np.diag(cm)
    # FN=cm.sum(axis=1)-np.diag(cm)
    # TP=np.diag(cm)
    # TN=cm.sum()-(FP+FN+TP)

    # print('False positive : \n {}'.format(FP))
    # print('False negative : \n {}'.format(FN))
    # print('True positive : \n {}'.format(TP))
    # print('True negative : \n {}'.format(TN))


    # TPR=TP/(TP+FN)
    # print('Sensitivity : {}\n'.format(TPR))
    # FNR=TN/(TN+FP)
    # print('Specificity : {}\n'.format(TPR))


    # Precision=TP/(TP+FP)
    # Accuracy=(TP+FP)/(TP+FP+TN+FN)
    # Recall=TP/(TP+FN)
    # FScore=2*(Precision*Recall)/(Precision+Recall)
    # print('Precision : {}\n'.format(Precision))
    # print('Accuracy : {}\n'.format(Accuracy))
    # print('Recall : {}\n'.format(Recall))
    

   
    ScienceAndTech=pd.DataFrame(columns=data_test.columns)
    Health=pd.DataFrame(columns=data_test.columns)
    entertainment=pd.DataFrame(columns=data_test.columns)
    business=pd.DataFrame(columns=data_test.columns)
    
    i=0
    for x in y_pred:
        row=data_test.iloc[i,:]
        if x==0:
            business=business.append(row)
        elif x==1:
            entertainment=entertainment.append(row)
        elif x==2:
            Health=Health.append(row)
        elif x==3:
            ScienceAndTech=ScienceAndTech.append(row)
        i=i+1
        
        
    ScienceAndTech['title']= ScienceAndTech['title'].str.capitalize()
    Health['title']=Health['title'].str.capitalize()
    entertainment['title']=entertainment['title'].str.capitalize()
    business['title']=business['title'].str.capitalize()  
    ScienceAndTech=ScienceAndTech.reset_index(drop=True)
    Health=Health.reset_index(drop=True)
    entertainment=entertainment.reset_index(drop=True)
    business=business.reset_index(drop=True)
   

    ScienceAndTech=ScienceAndTech.to_dict("records")
    collection1 = db['ScienceAndTech'] 
    collection1.drop()
    collection1 = db['ScienceAndTech'] 
    collection1.insert_many(ScienceAndTech)
    Health=Health.to_dict("records")
    collection2 = db['Health'] 
    collection2.drop()
    collection2 = db['Health'] 
    collection2.insert_many(Health)
    entertainment=entertainment.to_dict("records")
    collection3= db['entertainment'] 
    collection3.drop()
    collection3= db['entertainment'] 
    collection3.insert_many(entertainment)
    business=business.to_dict("records")
    collection4 = db['business']
    collection4.drop()
    collection4 = db['business']
    collection4.insert_many(business)   
    return render_template('index.html',context=mylist)
@app.route('/category/<name>')
def category(name):
    if name=='business':
        cursor=db.business.find()
    elif name=='Health':
        cursor=db.Health.find()
    elif name=='entertainment':
        cursor=db.entertainment.find()
    elif name=='ScienceAndTech':
        cursor=db.ScienceAndTech.find()
    list_cur = list(cursor)
    json_data= dumps(list_cur) 
    
    data=json.loads(json_data)
   
    desc = []
    news = []
    img = []
    url=[]
    category_name=name
    for i in range(len(data)):
        news.append(data[i]['title'])
        desc.append(data[i]['description'])
        img.append(data[i]['urlToImage'])
        url.append(data[i]['url'])
    mylist = zip(news, desc, img,url)
    return render_template('show.html', context = mylist,Heading=category_name)   
@app.route('/about_us')
def about_us():
    return render_template('index_about.html')
@app.route('/contact_us')
def contact_us():
    return render_template('index_contact_us.html')
@app.route('/contact_us',methods=["POST"])
def contact_us_():
    contactUs=pd.DataFrame(columns=['name','phone','email','subject','messages'])
   
    name_=request.form['name']
    phone_=request.form['phone']
    email_=request.form['email']
    subject_=request.form['subject']
    messages_=request.form['message']
    
        
    contactUs=contactUs.append({'name':name_,'phone':phone_,'email':email_,'subject':subject_,'messages':messages_}, ignore_index=True)
    print(contactUs.head)
    contactUs=contactUs.to_dict("records")
    print(contactUs)
    contactUs_colle = db2['contactUsDb'] 
    # contactUs_colle.insert_many(contactUs)
    print(contactUs_colle.insert_many(contactUs))
    print(contactUs_colle)
    
    return render_template('index.html')       
if __name__ == '__main__':
    app.run(debug = True)

 

# %%
