from os import system
import numpy as np
import pandas as pd
DIGITS = 	 [[1, 1, 1, 1, 1, 1, 0], # digit '0'
			  [0, 1, 1, 0, 0, 0, 0], # digit '1'
			  [1, 1, 0, 1, 1, 0, 1], # digit '2'
			  [1, 1, 1, 1, 0, 0, 1], # digit '3'
			  [0, 1, 1, 0, 0, 1, 1], # digit '4'
			  [1, 0, 1, 1, 0, 1, 1], # digit '5'
			  [1, 0, 1, 1, 1, 1, 1], # digit '6'
			  [1, 1, 1, 0, 0, 0, 0], # digit '7'
			  [1, 1, 1, 1, 1, 1, 1], # digit '8'
			  [1, 1, 1, 1, 0, 1, 1]] # digit '9'
N = 14
LABELS = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7']
# def corrupter(d, digits, n_size, pr_sf=0.1):
	
# 	data = np.zeros((n_size, N))	# Initialize
# 	for n in range(n_size):
# 		result = np.copy(digits)	# Take the seven clean segments
# 		for s, segment in enumerate(result):
# 			s_failure = np.random.uniform()
# 			if s_failure < pr_sf and result[s] == 1: # The segments ON can be switched off
# 				result[s] = 0
# 		data[n][:] = np.concatenate((np.array(digits),result))
# 	return data

def corrupter3(d, digits, n_size, pr_on = 0.7, pr_off = 0.7):
	data = np.zeros((n_size, N))	# Initialize
	for n in range(n_size):
		result = np.copy(digits)	# Take the seven clean segments
		all_zeros = True
		for s, segment in enumerate(result):
			o_failure = np.random.uniform()
			if segment == 0 :  			#segment off
				if o_failure > pr_off:
					result[s] = 1
			else:

				if o_failure > pr_on :  #neighbor on
					neighbor = s+1
					if neighbor == 7:
						neighbor = 0
					result[neighbor] =  1	

		data[n][:] = np.concatenate((np.array(digits),result))
	return data

def corrupter2(d, digits, n_size, pr_sf=0.3):
	
	data = np.zeros((n_size, N))	# Initialize
	for n in range(n_size):
		result = np.copy(digits)	# Take the seven clean segments
		all_zeros = True
		for s, segment in enumerate(result):

			s_failure = np.random.uniform()
			
			if s_failure < pr_sf : 
				if segment == 1 and not (all(v == 0 for v in result)):  #Verifies if we have all segments off
					result[s] = 0
				else:
					result[s] = 1
		data[n][:] = np.concatenate((np.array(digits),result))
	return data


# What about this: for each segment there is a probability p (say, 0.1) of the observed value being different than the true value of the segment; however we never have all segments off
# @Alessandro Antonucci would that work for having the sdd?


def generate_data_set2(train_size, test_size, file_name):

	# prepare training and testing set
	x_train = np.zeros((train_size * 10, N))
	x_test = np.zeros((test_size * 10, N))

	# generate stratified data
	for i, l in enumerate(DIGITS):
		data = corrupter3(i, l, train_size + test_size)
		x_train[i * train_size:(i + 1) * train_size] = data[test_size:]
		x_test[i * test_size:(i + 1) * test_size] = data[:test_size]
		
	np.savetxt(file_name + '100_n_train.csv', x_train, delimiter=',', fmt='%d', header=','.join(LABELS), comments='')
	np.savetxt(file_name + '100_n_test.csv', x_test, delimiter=',', fmt='%d', header=','.join(LABELS), comments='')

base_file_name = '../Data/seven_segments'
# n_train = 10
# n_test = 1
generate_data_set2(10,1,base_file_name)
