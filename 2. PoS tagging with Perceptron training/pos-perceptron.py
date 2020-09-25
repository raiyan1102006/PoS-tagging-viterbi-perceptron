#!/usr/bin/python3

import sys
import numpy as np            

def viterbi(x, emit, transmit, all_tags): #same as hw1
	#initialize delta table with 
	delta = np.ones((len(all_tags),len(x)))*-1*np.inf 

	#initialize back pointers with zeros
	backptr = np.zeros((len(all_tags),len(x)))
	
	for word_idx in range(len(x)):
		if word_idx==0: #first word in sentence
			for tag_idx in range(len(all_tags)): #consider only emission weight since it is first word
				delta[tag_idx][word_idx] = emit[(all_tags[tag_idx],x[word_idx])] if (all_tags[tag_idx],x[word_idx]) in emit else 0
			continue
		
		# from second word onwards
		for tag_idx in range(len(all_tags)): #has emission and transmission weights
			emission_value = emit[(all_tags[tag_idx],x[word_idx])] if (all_tags[tag_idx],x[word_idx]) in emit else 0
			
			for tag_idx_prev in range(len(all_tags)):
				transition_value = transmit[(all_tags[tag_idx_prev],all_tags[tag_idx])] if (all_tags[tag_idx_prev],all_tags[tag_idx]) in transmit else 0
				temp_delta_val = delta[tag_idx_prev][word_idx-1]+emission_value+transition_value
				if temp_delta_val>delta[tag_idx][word_idx]: #get the max value
					delta[tag_idx][word_idx] = temp_delta_val
					backptr[tag_idx][word_idx] = tag_idx_prev #keep a backward pointer

	# navigate backward pointer to generate the sequence y_hat
	current_idx = np.argmax(delta[:,-1])
	y_hat_idx = [current_idx]
	for j in range(np.shape(backptr)[1]-1,0,-1):  
		y_hat_idx.append(int(backptr[current_idx][j]))
		current_idx = int(backptr[current_idx][j])
	y_hat = [all_tags[y_hat_idx[k]] for k in range(len(y_hat_idx)-1,-1,-1)]

  
	return y_hat, np.amax(delta)


if __name__ == '__main__':

	emit = {} # emission weights
	transmit = {} # transmission weights
	
	# load training file
	print("Loading training file") 

	with open('train') as training_data:
		
		#generate all tags
		all_tags = [] # list of tags or states
		all_data = []
		for a_line in training_data:
			a_line = a_line.strip().split(' ')
			a_line = a_line[1:]
			all_data.append(a_line)
			x = a_line[0::2]
			y = a_line[1::2]
			
			# collecting all tags
			for yi in y:
				if yi not in all_tags:
					all_tags.append(yi)
			
			# doing an initial collection of emission and transmission weights. This helps
			# improve the performance right from the first iteration
			for i in range(len(y)):
				if (y[i],x[i]) not in emit:
					emit[(y[i],x[i])]=1
				if i>0:
					if (y[i-1],y[i]) not in transmit:
						transmit[(y[i-1],y[i])]=1
			

	print("Training file loaded\n")    

	# perceptron algorithm
	print("Running perceptron")
	n_itr = 10

	for itr in range(n_itr):
		print("Iteration "+str(itr+1))
		
		# learn from training data
		total_count_train = []
		correct_count_train = []
		
		for line_idx, a_line in enumerate(all_data):
			if line_idx%1000==0:
				print("Current training sentence ID "+str(line_idx))
			
			x = a_line[0::2]
			y = a_line[1::2]

			y_hat,amax = viterbi(x,emit,transmit,all_tags)

			total_count_train.append(len(y))
			correct_count_train.append(np.sum(np.array(y)==np.array(y_hat)))
			
			if y!=y_hat: #update weights
				for i in range(len(y)):

					emit[(y[i],x[i])] += 1
					
					if (y_hat[i],x[i]) not in emit:
						emit[(y_hat[i],x[i])] = -1 
					else:
						emit[(y_hat[i],x[i])] -= 1
						
					if i>0:

						transmit[(y[i-1],y[i])] += 1
						
						if (y_hat[i-1],y_hat[i]) not in transmit:
							transmit[(y_hat[i-1],y_hat[i])] = -1 
						else:
							transmit[(y_hat[i-1],y_hat[i])] -= 1
							
		accuracy_train = float(np.sum(correct_count_train))/float(np.sum(total_count_train))
		
		print("Training set accuracy: "+str(accuracy_train))
		
		
		#check accuracy in dev data 
		with open('dev') as dev_file:

			total_count_dev = []
			correct_count_dev = []
			i=0
			for a_line in dev_file:
				if i%1000==0:
					print("Current dev sentence ID "+str(i))
				a_line = a_line.strip().split(' ')
				a_line = a_line[1:]
				x = a_line[0::2]
				y = a_line[1::2]

				y_hat,amax = viterbi(x,emit,transmit,all_tags)

				total_count_dev.append(len(y))
				correct_count_dev.append(np.sum(np.array(y)==np.array(y_hat)))
				i+=1

		accuracy_dev = float(np.sum(correct_count_dev))/float(np.sum(total_count_dev))   
		print("Dev set accuracy: "+str(accuracy_dev))
		print(" ")


	# Saving weights to file
	print("Saving weights\n")
	f= open('trained_weights',"w")
	for key,value in emit.items():
		f.write("E_"+key[0]+"_"+key[1]+" "+str(value)+"\r\n")
	for key,value in transmit.items():
		f.write("T_"+key[0]+"_"+key[1]+" "+str(value)+"\r\n")
	f.close()
	

	# Report accuracy in test data
	print("Calculating accuracy in test data")
	with open('test') as test_file:
		total_count_test = []
		correct_count_test = []

		i=0
		for a_line in test_file:
			if i%1000==0:
				print("Current test sentence ID "+str(i))
			a_line = a_line.strip().split(' ')
			a_line = a_line[1:]
			x = a_line[0::2]
			y = a_line[1::2]

			y_hat,amax = viterbi(x,emit,transmit,all_tags)

			total_count_test.append(len(y))
			correct_count_test.append(np.sum(np.array(y)==np.array(y_hat)))
			i+=1


	accuracy_test = float(np.sum(correct_count_test))/float(np.sum(total_count_test))

	print("Test set accuracy: "+str(accuracy_test))
			
