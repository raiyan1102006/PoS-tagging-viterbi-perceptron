# Parts of
======

Homework 1
Raiyan Abdul Baten


Usage Instruction
=================

1. To run the sample test case given on the website, please execute the following command:

python3 hw1.py cat_weights cat_data testcase

Here, cat_weights and cat_data are saved as input files in the submission folder. The 'testcase' argument makes sure the output is properly formatted.


2. To find the performance on the train.weight and test files, please execute the following command:

python3 hw1.py train.weights test

Here, no 'testcase' argument is needed.


Performance
===========
For the sample test case (cat_weights and cat_data), the program generates the expected output.

For the train.weight and test files, the achieved accuracy is 95.44%. However, this performance can only be obtained if the given weights are shifted by roughly more than 15--- in my implementation, I've shifted the weights by 30. Without the shifting, the accuracy is very poor. For the missing values, I've used a weight of 0. The rest of the algorithm is commented inline.
