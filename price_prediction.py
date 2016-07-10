import gc
import glob
import numpy as np
import os
import pandas
import pickle
import sklearn.metrics
import time
import httplib2, json
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient import discovery
from googleapiclient.errors import HttpError
from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor
from itertools import chain, combinations

feature_vector  = ['unblendedcost', 'productname', 'ys_usagetype', 'ys_usagetypegroup', 'usagequantity']
service_account = 'service_account.json' 
project_id      = 'api-project-336407747088'

def get_google_prediction_api():
	scope = [
		'https://www.googleapis.com/auth/prediction',
		'https://www.googleapis.com/auth/devstorage.read_only'
	]
	credentials = ServiceAccountCredentials.from_json_keyfile_name(service_account, scopes=scope)
	http = credentials.authorize(httplib2.Http())
	return discovery.build("prediction", "v1.6", http=http)
	
