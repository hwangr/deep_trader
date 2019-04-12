import gzip
import os

import numpy as np
import six
from six.moves.urllib import request

from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import dateutil.parser
import pdb
import glob
import pickle
import shelve
import six
from six.moves.urllib import request
import hashlib
import json


episode = 10 #lenght of one episode
data_array = []
parent_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
#raw_data_file  = os.path.join(parent_dir,'tensor-reinforcement/NIFTY50.csv') 
path  = os.path.join(parent_dir,'ib/csv_data/AJP.csv') 
moving_average_number = 1000 #number of time interval for calculating moving average
#pdb.set_trace()

def prepare_data(path):
#	stock_data = genfromtxt(raw_data_file, delimiter=',', dtype=None, names=True)	
	stock_data = genfromtxt(path, delimiter=',', dtype=None, names=True)
	average_dataset = []
	total_data = []
	temp_episode = []
	data_dict = {}
	index = 0
	lines = []
	for data in stock_data:
		temp = [data[2], data[3], data[4], data[5],data[8]]
		average_dataset.append(temp)
		print(index)
		print(len(average_dataset))
		if index > moving_average_number:
			mean = find_average(average_dataset)
			mean_array = average_dataset/mean
			last_one_hour_average = find_average(mean_array[-60:])
			last_one_day_average = find_average(mean_array[-300:])
			last_3_day_average = find_average(mean_array[-900:]) #this might change
			last_minute_data = mean_array[-1]
			average_dataset = average_dataset[1:]
			vector = []
			vector.extend(last_minute_data)
			vector.extend(last_one_hour_average)
			vector.extend(last_one_day_average)
			vector.extend(last_3_day_average)
			average_price = sum(temp[0:-2]) / float(len(temp[0:-2]))
			dict_vector = temp + [average_price]
			#data_dict[list_md5_string_value(vector)] = dict_vector
			md5 = list_md5_string_value(vector)
			lines.append("key:{0}\nmd5:{1}\nvalue:{2}".format(vector, md5, dict_vector))
			data_dict[md5] = dict_vector
			total_data.append(vector)
		index += 1
	with open("data.pkl", "wb") as myFile:
		#six.moves.cPickle.dump(total_data, myFile, -1)
		pickle.dump(total_data, myFile, -1)
	print("Done")
	with open("data_dict.pkl","wb") as myFile:
		#six.moves.cPickle.dump(data_dict, myFile, -1)
		pickle.dump(data_dict, myFile, -1)

def find_average(data):
    return np.mean(data, axis=0)

def load_data(file,episode):
	data = load_file_data(file)
	return list(map(list,zip(*[iter(data)]*episode)))

def load_file_data(file):
    with open(file, 'rb') as myFile:
		#data = six.moves.cPickle.load(myFile, encoding='latin-1')
         data = pickle.load(myFile, encoding='latin-1')
    return data

def list_md5_string_value(list):
    string = json.dumps(list)
    return hashlib.md5(string.encode('utf-8')).hexdigest()

#prepare_data(path)