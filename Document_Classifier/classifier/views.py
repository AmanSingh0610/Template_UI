import json
from django.shortcuts import render,redirect
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .models import Results
from django.views import View
# Imports.
import numpy
import docx
import pdfplumber
import tensorflow as tf
import pickle
from tensorflow import keras
from keras import layers, models, optimizers
from keras.models import load_model
from keras.preprocessing import text, sequence
import pandas, numpy, string
import pandas as pd
import nltk
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('omw-1.4')
import os
import re, unidecode
from bs4 import BeautifulSoup
from nltk.stem.porter import PorterStemmer 
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import ibm_boto3
from ibm_botocore.client import Config, ClientError
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import datetime
import hashlib
import hmac
import requests
from requests.utils import quote
from Document_Classifier.settings import ibm_api_key_id,cos,cos2

#final Model

final_model = pickle.load(open('ML_model/SVM2', 'rb'))
Tfidf_vect = pickle.load(open('ML_model/Tfidf_vect', 'rb'))
Encoder = pickle.load(open('ML_model/Encoder', 'rb'))
# load tokenizer
tokenizer = text.Tokenizer()
with open('ML_model/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


# Needed only once
def remove_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text(separator=" ")
    return stripped_text
def remove_accented_chars(text):
    text = unidecode.unidecode(text)
    return text
def remove_numbers(text): 
    result = re.sub(r'\d+', '', text) 
    return result
def remove_newline(text):
    return text.replace('\n', " ")    
def remove_slash_with_space(text): 
    return text.replace('\\', " ")
def remove_punctuation(text): 
    translator = str.maketrans('', '', string.punctuation) 
    return text.translate(translator) 
def text_lowercase(text): 
    return text.lower()     
def remove_whitespace(text): 
    return  " ".join(text.split()) 
def remove_stopwords(text): 
    stop_words = set(stopwords.words("english")) 
    word_tokens = word_tokenize(text) 
    filtered_text = [word for word in word_tokens if word not in stop_words] 
    return ' '.join(filtered_text)
def stem_words(text): 
    stemmer = PorterStemmer() 
    word_tokens = word_tokenize(text) 
    stems = [stemmer.stem(word) for word in word_tokens] 
    return ' '.join(stems)
def lemmatize_words(text): 
    lemmatizer = WordNetLemmatizer() 
    word_tokens = word_tokenize(text) 
    # provide context i.e. part-of-speech 
    lemmas = [lemmatizer.lemmatize(word, pos ='v') for word in word_tokens] 
    return ' '.join(lemmas) 
# hashing methods
def hash(key, msg):
    return hmac.new(key, msg.encode('utf-8'), hashlib.sha256).digest()

# region is a wildcard value that takes the place of the AWS region value
# as COS doen't use regions like AWS, this parameter can accept any string


def multi_part_upload(bucket_name, item_name, file_path):
    try:
        print("Starting file transfer for {0} to bucket: {1}\n".format(item_name, bucket_name))
        # set 5 MB chunks
        part_size = 1024 * 1024 * 5

        # set threadhold to 15 MB
        file_threshold = 1024 * 1024 * 15

        # set the transfer threshold and chunk size
        transfer_config = ibm_boto3.s3.transfer.TransferConfig(
            multipart_threshold=file_threshold,
            multipart_chunksize=part_size
        )
       
        # the upload_fileobj method will automatically execute a multi-part upload
        # in 5 MB chunks for all files over 15 MB
        with open(file_path, "rb") as file_data:
            cos.Object(bucket_name, item_name).upload_fileobj(
                Fileobj=file_data,
                Config=transfer_config
            )
        print("Transfer for {0} Complete!\n".format(item_name))
    except ClientError as be:
        print("CLIENT ERROR: {0}\n".format(be))
    except Exception as e:
        print("Unable to complete multi-part upload: {0}".format(e))

        
def delete_item(bucket_name, object_name):
    try:
        cos2.delete_object(Bucket=bucket_name, Key=object_name)
        print("Item: {0} deleted!\n".format(object_name))
    except ClientError as be:
        print("CLIENT ERROR: {0}\n".format(be))
    except Exception as e:
        print("Unable to delete object: {0}".format(e))

def list_files(bucket_name,max_keys=10):
    print("Retrieving bucket contents from: {0}".format(bucket_name))
    try:
        more_results = True
        next_token = ""
        while (more_results):
            response = cos2.list_objects_v2(Bucket=bucket_name, MaxKeys=max_keys,ContinuationToken=next_token)
            files = response["Contents"]
            # for file in files:
            #     print("Item: {0} ({1} bytes).".format(file["Key"], file["Size"]))
            if (response["IsTruncated"]):
                next_token = response["NextContinuationToken"]
                print("...More results in next batch!\n")
            else:
                more_results = False
                next_token = ""
    except ClientError as be:
        print("CLIENT ERROR: {0}\n".format(be))
    except Exception as e:
        print("Unable to retrieve bucket contents: {0}".format(e))
    return files


def download_item (item_key, save_to, bucket_name):
    cos = ibm_boto3.client("s3",
        ibm_api_key_id="NoYjdUF95eIFA4aYZBT0hv5PNqdKUVAU_RtvBrz2ucCu",
        ibm_service_instance_id="crn:v1:bluemix:public:iam-identity::a/08ce398b923b4c93a5bf90c9401144b4::serviceid:ServiceId-2ec0216f-265e-4b6d-aab3-39eca6f7d10e",
        config=Config(signature_version="oauth"),
        endpoint_url="https://s3.jp-tok.cloud-object-storage.appdomain.cloud"
        )
    print("Downloading {0} from {1}".format(item_key, bucket_name))
    try:
        res=cos.download_file(Bucket=bucket_name, Key=item_key, Filename=save_to)
        print("Downloaded {0} to {1}".format(item_key, save_to))
        return True
    except ClientError as be:
        print("CLIENT ERROR: {0}\n".format(be))
    except Exception as e:
        print("Unable to Download from Bucket: {0}".format(e))
    return False

def download_item_url():
    access_key = "d5e8d33ca16a4bf6ba4770b055914f79"
    secret_key = "fd71d42f9d19b82010bb571b2f25d51fd3cd087153a3198b"
    # request elements
    http_method = 'GET'
    region = "jp-tok"
    bucket = "filesstoragebucket"
    cos_endpoint ="https://s3.tok.ap.cloud-object-storage.appdomain.cloud"
    host = cos_endpoint
    endpoint = 'https://'+ "s3.jp-tok.cloud-object-storage.appdomain.cloud"
    object_key = "abc.txt"
    expiration = 86400  # time in seconds    
    
    # assemble the standardized request
    time = datetime.datetime.utcnow()
    timestamp = time.strftime('%Y%m%dT%H%M%SZ')
    datestamp = time.strftime('%Y%m%d')


    standardized_querystring = ('X-Amz-Algorithm=AWS4-HMAC-SHA256' +
                            '&X-Amz-Credential=' + access_key + '/' + datestamp + '/' + region + '/s3/aws4_request' +
                            '&X-Amz-Date=' + timestamp +
                            '&X-Amz-Expires=' + str(expiration) +
                            '&X-Amz-SignedHeaders=host')
    standardized_querystring_url_encoded = quote(standardized_querystring, safe='&=')
    
    standardized_resource = '/' + bucket + '/' + object_key
    standardized_resource_url_encoded = quote(standardized_resource, safe='&')
    
    payload_hash = 'UNSIGNED-PAYLOAD'
    standardized_headers = 'host:' + host
    signed_headers = 'host'
    
    standardized_request = (http_method + '\n' +
                            standardized_resource + '\n' +
                            standardized_querystring_url_encoded + '\n' +
                            standardized_headers + '\n' +
                            '\n' +
                            signed_headers + '\n' +
                            payload_hash).encode('utf-8')

    # assemble string-to-sign
    hashing_algorithm = 'AWS4-HMAC-SHA256'
    credential_scope = datestamp + '/' + region + '/' + 's3' + '/' + 'aws4_request'
    sts = (hashing_algorithm + '\n' +
        timestamp + '\n' +
        credential_scope + '\n' +
        hashlib.sha256(standardized_request).hexdigest())
    
    # generate the signature
    signature_key = createSignatureKey(secret_key, datestamp, region, 's3')
    signature = hmac.new(signature_key,
                        (sts).encode('utf-8'),
                        hashlib.sha256).hexdigest()
    
    # create and send the request
    # the 'requests' package autmatically adds the required 'host' header
    request_url = (endpoint + '/' +
                bucket + '/' +
                object_key + '?' +
                standardized_querystring_url_encoded +
                '&X-Amz-Signature=' +
                signature)

    print(('request_url: %s' % request_url))
    print(('\nSending `%s` request to IBM COS -----------------------' % http_method))
    print(('Request URL = ' + request_url))
    request = requests.get(request_url)
    
    print('\nResponse from IBM COS ---------------------------------')
    print(('Response code: %d\n' % request.status_code))
    print((request.text))
    
    # this information can be helpful in troubleshooting, or to better
    # understand what goes into signature creation
    print('These are the values used to construct this request.')
    print('Request details -----------------------------------------')
    print(('http_method: %s' % http_method))
    print(('host: %s' % host))
    print(('region: %s' % region))
    print(('endpoint: %s' % endpoint))
    print(('bucket: %s' % bucket))
    print(('object_key: %s' % object_key))
    print(('timestamp: %s' % timestamp))
    print(('datestamp: %s' % datestamp))
    print('Standardized request details ----------------------------')
    print(('standardized_resource: %s' % standardized_resource_url_encoded))
    print(('standardized_querystring: %s' % standardized_querystring_url_encoded))
    print(('standardized_headers: %s' % standardized_headers))
    print(('signed_headers: %s' % signed_headers))
    print(('payload_hash: %s' % payload_hash))
    print(('standardized_request: %s' % standardized_request))
    print('String-to-sign details ----------------------------------')
    print(('credential_scope: %s' % credential_scope))
    print(('string-to-sign: %s' % sts))
    print(('signature_key: %s' % signature_key))
    print(('signature: %s' % signature))
    print('Because the signature key has non-ASCII characters, it is necessary to create a hexadecimal digest for the purposes of checking against this example.')
    signature_key_hex = createHexSignatureKey(secret_key, datestamp, region, 's3')
    print(('signature_key (hexidecimal): %s' % signature_key_hex))
    print('Header details ------------------------------------------')
    print(('pre-signed URL: %s' % request_url))
    return{
        "status code":200,
        "headers":{
            "Content-Type":"application/json",
        },
        "body": json.dumps({"URL":request_url})
    }

def createSignatureKey(key, datestamp, region, service):
    keyDate = hash(('AWS4' + key).encode('utf-8'), datestamp)
    keyRegion = hash(keyDate, region)
    keyService = hash(keyRegion, service)
    keySigning = hash(keyService, 'aws4_request')
    return keySigning
    
def hex_hash(key, msg):
    return hmac.new(b'key', msg.encode('utf-8'), hashlib.sha256).hexdigest()

def createHexSignatureKey(key, datestamp, region, service):
    keyDate = hex_hash(('AWS4' + key).encode('utf-8'), datestamp)
    keyRegion = hex_hash(keyDate, region)
    keyService = hex_hash(keyRegion, service)
    keySigning = hex_hash(keyService, 'aws4_request')
    return keySigning


# def download_url_():
#     cos = ibm_boto3.client('s3', 
#         "WPgQ25hobq4c2VmqU3w8gL5gtvYjT25PvdS7c8v2K4zL",
#         endpoint_url="https://s3.tok.ap.cloud-object-storage.appdomain.cloud", 
#         aws_access_key_id="d5e8d33ca16a4bf6ba4770b055914f79", 
#         aws_secret_access_key= "fd71d42f9d19b82010bb571b2f25d51fd3cd087153a3198b")

#     theURL=cos.generate_presigned_url('get_object', Params = {'Bucket':"filesstoragebucket" , 'Key': "abc.txt"}, ExpiresIn = 600)
#     print(theURL)

def show_file(endpoint,bucket,object):
    url="https://"+endpoint+"/"+bucket+"/"+object
    return url
    
# Perform preprocessing
def perform_preprocessing(text):
    text = remove_html_tags(text)
    text = remove_accented_chars(text)
    text = remove_numbers(text)
    text = remove_newline(text)
    text = remove_stopwords(text)
    text = text_lowercase(text)
    text = remove_slash_with_space(text)
    text = remove_punctuation(text)
    # text = stem_words(text)
    text = lemmatize_words(text)
    text = remove_whitespace(text)
    return text

def tfidf_weights(text):
    Tfidf_vect_filter = pickle.load(open('ML_model/Tfidf_vect_filter', 'rb'))
    Content_Tfidf = Tfidf_vect_filter.transform([text])
    df_tfidf= pd.DataFrame(Content_Tfidf.T.todense(), index=Tfidf_vect_filter.get_feature_names(), columns=["Tfidf Weights"])
    return ' '.join(df_tfidf.sort_values(by=["Tfidf Weights"],ascending=False)[:40].index)  

def prediction(text,file):
    #inp = [perform_preprocessing(x) for x in inp]
    #inp_token = tokenizer.texts_to_sequences(inp)
    text = perform_preprocessing(text)
    text_vectorized=Tfidf_vect.transform([text])
    prediction_SVM = final_model.predict(text_vectorized.toarray())
    print('Confidence Score: ',final_model.predict_proba(text_vectorized.toarray()))
    print(f"Prediction {file}: ",Encoder.inverse_transform(prediction_SVM)[0])
    return Encoder.inverse_transform(prediction_SVM)[0]

class Start(View):
    def get(self, request):
        return redirect('/index')

class History(View):
    def get(self, request):
        #results = Results.objects.all()
        results=list_files("filesstoragebucket")
        return render(request,'history.html',{'results':results,'meth':True})
    
    def post(self, request):
        for i in request.POST.keys():
            print(i)
        id=int(i)
        results=list_files("filesstoragebucket")
        selected_result=results.get(id==id)
        #selected_result = Results.objects.get(id=id)
        #results = Results.objects.all()
        
        return render(request,'history.html',{'selected_result':selected_result,'meth':False,'results':results}) 
class Index(View):
    def get(self, request):
        print('called',request.FILES)
        return render(request,'index.html')

    def post(self, request):
        print(request.FILES)
        if request.FILES['file1']:
            result = Results()
            class_dict = {'0': 'AI And Analytics', '1': 'Automation','2':'Blockchain','3':'Connected Operations','4':'Data Transformation','5':'Microsoft'}
            dic_typeof = {0:'MSA',1:'SOW',2:'RFP'}
            maxlen = 300
            confidence = 92
            for i in request.POST:
                print(i)
                if i == 'csrfmiddlewaretoken':
                    continue
                if i == 'doctype':
                   doctype = request.POST['doctype']
            #     if i == 'savefiles':
            #         savefiles = request.POST['savefiles']
            # print(savefiles)
        #Recieving the uploaded file and saving it in the Bucket
            fileobj=request.FILES['file1'] 
            fs=FileSystemStorage()
            filepathname=fs.save(fileobj.name,fileobj)
            multi_part_upload("filesstoragebucket",fileobj.name,r"C:\Users\AVVBGV744\Downloads\runcommand.txt")
            #delete_item("filesstoragebucket","abc.txt")
            download_item_url()
            filepathname=fs.url(filepathname)
            filepath = os.path.basename(fileobj.name)
            testfile='./media/'+str(fileobj.name)

        #Based on File type different kind of extraction will be carried out. 
            if doctype=='pdf':
                file = pdfplumber.open(testfile)
                text = ' '.join([page.extract_text() for page in file.pages])
            elif doctype=='text':
                with open(testfile,'r',encoding = 'utf-8') as f:
                    text = f.read()
            elif doctype=='docx':
                doc = docx.Document(testfile)
                fullText = []
                for para in doc.paragraphs:
                    fullText.append(para.text)
                text = '\n'.join(fullText)
            
            label = str(prediction(text,fileobj.name))
        
            typeof = 0
            typeof = dic_typeof[typeof]
            keywords = ["data","technology","aritificial intelligence","pipeline"]
            kw = ",".join(keywords)
            result.docname = fileobj.name
            print(label)
            result.confidence = confidence
            result.ddt = True
            result.keywords = kw
            result.filepath = fileobj
            result.save()
            
            #download_url_new()
            #download_item ("test_-_Copy_2.txt", r"C:\Users\AVVBGV744\Downloads\abc.txt", "file-storage-bucket2")
            redirected = '/detect/'+str(result.id)
            return redirect(redirected)
            #return render(request,'detect.html',{'label':label,'index':predicted_label[0],'keywords':keywords,'typeof':typeof,'lol':"92"})  

class Detect(View):
    def get(self, request,id):
        current_result = Results.objects.get(id=id)  
        return render(request,"detect.html",{'current_result':current_result}) 
