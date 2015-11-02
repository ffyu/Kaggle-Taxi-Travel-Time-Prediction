# Kaggle-Taxi-Travel-Time-Prediction

https://www.kaggle.com/c/pkdd-15-taxi-trip-time-prediction-ii

**Final Leaderboard Ranking: 48/345**

This is one of the models used in my final submission of the Kaggle's Taxi trip time prediction competition. 

The basic idea of a Random Forest model is to ensemble a number of different trees, with each tree trained on a bootstrapped sample and a subset of features under each split. If we decide to go with a large number of trees in the forest for a sizable training set, the total memory required in the process would be huge. With the constraint of RAM on a personal laptop (normally 8-16 GB), this can cause difficulties in training the model.

The script **Submission.py** utilizes a specially designed Scalable Random Forecast algorithm. The idea is similar to the original Random Forest - i.e. to generate diversified trees and average the results to improve performance. To scale the problem, instead of using a bootstrapped sample with the same size as the original data set, we split the entire training set into a number of partitions, and train each single tree on a single partition. The number of partitions is equal to the number of trees and each partition has the same size. Now we have a much smaller training data for each tree but still manage to keep the sample diversified. The idea of subsetting the features on every split is still kept in this adjusted algorithm. All those should lead to many different yet useful single trees. By bagging them, we would be able to achieve similar accuracy compared to traditional Random Forest, but with a much less requirement of RAM usage.

The dependencies of this script include:
 - [re](https://docs.python.org/2/library/re.html)
 - [numpy](http://docs.scipy.org/doc/numpy/user/install.html)
 - [pandas](http://pandas.pydata.org/)
 - [sqlite3](https://docs.python.org/2/library/sqlite3.html)
 - [datetime](https://docs.python.org/2/library/datetime.html)
 
To run the script, please change the global variable **FOLDER** to your local directory, and also download the train and test sets from the competition website. The script would generate a bunch of intermediate files. The final prediction of the test set is named **"submission_final.csv"**. The running time is roughly two hours on an i7-16G laptop.

Also note that this is only one model that contributes to my final result. I used another model which breaks down the training set by varying time of the day. That model is too lengthy to show it here. 
 
Please feel free to email me if you have any comments/suggestions.

Feifei Yu

feifeiyu1204@gmail.com
