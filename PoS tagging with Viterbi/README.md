# Parts of Speech Tagging using Viterbi Decoder

**Author:** Raiyan Abdul Baten

This project was done as part of a homework in the CSC448 Statistical Speech and Language Processing course at the University of Rochester.


## Usage Instruction

1. To run the sample test case given in the out.out file, please execute the following command:
```
python3 pos-viterbi.py cat_weights cat_data testcase
```
Here, cat_weights and cat_data are input files. The 'testcase' argument makes sure that the output is properly formatted.


2. To find the performance on the train.weight and test files, please execute the following command:
```
python3 pos-viterbi.py train.weights test
```
Here, no 'testcase' argument is needed.


## Performance
For the sample test case (cat_weights and cat_data), the program generates the expected output in out.out.

For the train.weight and test files, the achieved accuracy is 95.44%. However, this performance can only be obtained if the given weights are shifted by roughly more than 15--- in my implementation, I've shifted the weights by 30. Without the shifting, the accuracy is very poor. For the missing values, I've used a weight of 0. The rest of the algorithm is commented inline.
