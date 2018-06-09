# Fortune-Cookie-Classifier-using-Naive-Bayes

This mini project is a Naive Bayes classifier built in python classifies fortune cookie messages as a future prediction or a waise saying. The dataset used for this project is present in the data folder of this repository. The data folder contains 5 text files which contains following:
  1. **stoplist.txt** : Contains a list of stop words.
  2. **traindata.txt** : Training messages.
  3. **trainlabels.txt** : Train labels corresponding to each fortune cookie message (1 - future prediction and 0 - wise saying).
  4. **testdata.txt** : Test messages.
  5. **testlabels.txt** : Test labels corresponding to each fortune cookie message (1 - future prediction and 0 - wise saying).
  
Before running the program, make sure you have following packages installed or install them using following commands:
 ```python
 pip install sklearn
 pip install numpy
 pip install pandas
 pip install matplotlib
 ```
 
To run the program, open a terminal and run following command:
 ```python
 python bayes.py
 ```
