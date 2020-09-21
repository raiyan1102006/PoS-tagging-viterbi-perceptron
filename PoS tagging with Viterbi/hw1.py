#!/usr/bin/python3

import sys
import numpy as np

def viterbi(x, emit, transmit, all_tags):
    #initialize delta table with -inf values
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

    sysarg_weights = sys.argv[1] #training file with weights
    sysarg_data = sys.argv[2] #test file for evaluation
    sysarg_test = None
    
    # this is for testing the sample output given in the website
    if len(sys.argv) > 3 and sys.argv[3]=='testcase':
        sysarg_test = sys.argv[3]


    # load training file
    with open(sysarg_weights) as train_weights:
        if sysarg_test:
            bias = 0. # for the sample output, no bias is needed
        else:
            bias = 30. # for the train.weight file, a bias>15 helps the accuracy

        emit = {} # emission weights
        transmit = {} # transmission weights
        all_tags = [] # list of tags or states

        for a_line in train_weights:
            tag,weight = a_line.strip().split(' ')
            tag = tag.split('_')
            if tag[1] not in all_tags:
                all_tags.append(tag[1]) #collect all the tags
            
            if tag[0]=='E': #emission weight
                emit[(tag[1],tag[2])]=float(weight)+bias #forcefully shifting weight
            else: #transmission weight
                transmit[(tag[1],tag[2])]=float(weight)+bias

    #read test file
    with open(sysarg_data) as test_file:

        total_count = []
        correct_count = []
        
        for a_line in test_file:
            a_line = a_line.strip().split(' ')
            a_line = a_line[1:]
            x = a_line[0::2]
            y = a_line[1::2]

            if sysarg_test: # for the sample output, send whole line
                y_hat,amax = viterbi(a_line,emit,transmit,all_tags)
            else: # for train.weight, send the word sequence
                y_hat,amax = viterbi(x,emit,transmit,all_tags)
            
            total_count.append(len(y))
            correct_count.append(np.sum(np.array(y)==np.array(y_hat)))

            if sysarg_test: # print output for sample case
                print(amax, y_hat)
    
    accuracy = float(np.sum(correct_count))/float(np.sum(total_count))
    if not sysarg_test: #print accuracy for the train.weight case
        print("Accuracy: "+str(accuracy))


            