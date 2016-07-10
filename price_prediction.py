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

def train_model_at_google(full_training_path, model_id):
	api = get_google_prediction_api()
	api.trainedmodels().insert(project=project_id, body={
		'id': model_id,
		'storageDataLocation': full_training_path,
		'modelType': 'regression'
	}).execute()

def explained_variance(ground_truth, predicted):
	return sklearn.metrics.explained_variance_score(ground_truth, predicted)

def r2_score(ground_truth, predicted):
	return  sklearn.metrics.r2_score(ground_truth, predicted)
	
def mean_square(ground_truth, predicted):
	return sklearn.metrics.mean_squared_error(ground_truth, predicted)

def make_prediction(data_frame, model_id):
	api   = get_google_prediction_api()
	model = service.trainedmodels().get(project=project_id, id=model_id).execute()
	if model.get('trainingStatus') != 'DONE':
		print("Model is (still) training. \nPlease wait and run me again!") #no polling
		exit()
	print("Model is ready.")
	label = []
	for row in data_frame.itertuples(index = False):
		label.append(fetch_prediction_from_google(api, list(row), model_id))
		time.sleep(6)
	return label

def fetch_prediction_from_google(service, input_vector, model_id):
	prediction = service.trainedmodels().predict(project=project_id, id=model_id, body={ 'input': { 'csvInstance': input_vector } }).execute()
	return prediction.get('outputValue')

def string_to_integer(product_names):
	product_name_to_number = {}
	count 					   = 1
	for product_name in product_names:
		if product_name not in product_name_to_number:
			product_name_to_number[product_name] = count
			count += 1 
	return product_name_to_number

def get_feature_powerset(feature_columns):
	power_sets = []
	i = set(feature_columns[:-1])
	for power_set in chain.from_iterable(combinations(i, r) for r in range(len(i)+1)):
		power_sets.append(list(power_set))
	return power_sets[1:]
	
def train_gradient_boosting(data_type):
	df = pandas.read_csv('training_data/' + data_type + '/' + data_type + '.csv', delimiter = ',')
	predictor = model_training(df[feature_vector[1:]], df[feature_vector[1]])
	write_to_pickle('predictor/'+ data_type +'.pickle', predictor)
	del df 
	gc.collect()

def test_gradient_boosting(data_type):
	if not os.path.isfile('predictor/'+ data_type +'.pickle'):
		print('Model not available')
	else:
		df = pandas.read_csv('testing_data/' + data_type + '/' + data_type + '.csv', delimiter = ',')
		predictor = load_data_from_pickle_file('predictor/'+ data_type +'.pickle')
		return predictor.predict(df[feature[1:]])
		
def model_training(training_data, target_label):
	return GradientBoostingRegressor(verbose=True, n_estimators=100, learning_rate=0.1, max_depth=7, random_state=0, loss='ls').fit(training_data, target_label)
	
def write_to_pickle(file_name, data):
	with open(file_name, 'wb') as handle:
		pickle.dump(data, handle)

def load_data_from_pickle_file(filename):
    with open(filename, 'rb') as handle:
        words = pickle.load(handle)
    return words
def data_sample_size(total, testing_size):
	testing_size  = int((total * testing_size) / 100)
	training_size = total - testing_size
	return training_size, testing_size
		
def write_data_frames(data_frame, training_size, testing_size, training_path, testing_path):
	data_frame.sample(training_size).to_csv(training_path, sep = ",", index = False)
	data_frame.sample(testing_size).to_csv(testing_path, sep = ",", index = False)
	
def create_categorical_data(data_frame, testing_sample_size):
	training_size, testing_size = data_sample_size(data_frame.shape[0], testing_sample_size)
	write_data_frames(data_frame, training_size, testing_size, 'training_data/categorical_data/categorical_data.csv', 'testing_data/categorical_data/categorical_data.csv')
	
def prepare_numerical_data(data_frame):
	hash_of_product_names = load_data_from_pickle_file('hash_of_product_names.pickle') if os.path.isfile('hash_of_product_names.pickle') else string_to_integer(np.sort(data_frame['productname'].unique()))
	hash_of_usage_type	  = load_data_from_pickle_file('hash_of_usage_type.pickle') if os.path.isfile('hash_of_usage_type.pickle') else string_to_integer(np.sort(data_frame['ys_usagetype'].unique()))
	hash_of_usage_group	  = load_data_from_pickle_file('hash_of_usage_group.pickle') if os.path.isfile('hash_of_usage_type.pickle') else string_to_integer(np.sort(data_frame['ys_usagetype'].unique()))
	if not os.path.isfile('hash_of_product_names.pickle'):
		write_to_pickle('hash_of_product_names.pickle', hash_of_product_names)
		write_to_pickle('hash_of_usage_type.pickle', hash_of_usage_type)
		write_to_pickle('hash_of_usage_group.pickle', hash_of_usage_group)
	data_frame['productname'].replace(hash_of_product_names, inplace = True)
	data_frame['ys_usagetype'].replace(hash_of_usage_type, inplace = True)
	data_frame['ys_usagetypegroup'].replace(hash_of_usage_group, inplace = True)
	return data_frame
	
def create_numerical_data(data_frame, testing_sample_size):
	temp = prepare_numerical_data(data_frame.copy())
	training_size, testing_size = data_sample_size(temp.shape[0], testing_sample_size)
	write_data_frames(temp, training_size, testing_size, 'training_data/numerical_data/numerical_data.csv', 'testing_data/numerical_data/numerical_data.csv')
	del temp
	gc.collect()
	
def create_normalized_data(data_frame, testing_sample_size):
	temp   = prepare_numerical_data(data_frame.copy())
	scalar = preprocessing.RobustScaler()
	temp[feature_vector[1:]] = scalar.fit_transform(temp[feature_vector[1:]])
	training_size, testing_size = data_sample_size(temp.shape[0], testing_sample_size)
	write_data_frames(temp, training_size, testing_size, 'training_data/normalized_data/normalized_data.csv', 'testing_data/normalized_data/normalized_data.csv')	 
	del temp
	gc.collect()
		
def prepare_data(all_excel_files):
	data_frames 		  = []
	for files in all_excel_files:
		print(files)
		data_frames.append(pandas.read_csv(str(files), delimiter= '|', usecols = feature_vector)) 		
	raw_data 		 	  = pandas.concat(data_frames)
	del data_frames
	gc.collect()
	raw_data.fillna(0, inplace = True)
	return raw_data[feature_vector]

def create_training_data(**kwargs):
	testing_sample_size = kwargs.get('testing_sample_size') if 'testing_sample_size' in kwargs else 20
	training_files = glob.glob('data/*.csv')
	training_data  = prepare_data(training_files)
	create_categorical_data(training_data, testing_sample_size)
	create_numerical_data(training_data, testing_sample_size)
	create_normalized_data(training_data, testing_sample_size)
	del training_data
	gc.collect()
