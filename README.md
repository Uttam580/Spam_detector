 
# Spam Detection 

Simple spam detection app integarted with Flask and deployed over heroku.

URL : https://spam-detector20.herokuapp.com/


Project summary : 

==============================================================

The goal of this was to a spam detection model in python to automatically classify 
a message as either spam or ham(legitimate messages).
Hosting on Heroku using Flask (a lightweight web application framework).

===============================================================

# input 
![](input.PNG)

# output
![](output.PNG)



# Contents

* ```app.py``` - main file
* ```templates```  - This folder contains html for home and result page
* ```models``` - This folder contains predict script and pickle file for nlp model andCountVectorizer
* ```src```- This folderconatins source file for trainning the model.
* ```requirement.txt```- contains requirement file to run the model on cloud.
* ```proc file ``` -it  specifies the commands that are executed by the app on startup

#  Source  : 

Dataset: https://www.kaggle.com/uciml/sms-spam-collection-dataset

https://www.sciencedirect.com/science/article/pii/S2405844018353404

https://medium.com/@davedodea/how-to-deploy-a-python-app-to-heroku-12289912f29c





