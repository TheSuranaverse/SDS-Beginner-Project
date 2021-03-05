# Detecting Fake News with Python – Objective
To build a model to accurately classify a piece of news as REAL or FAKE.

# Detecting Fake News with Python – About the Python Project
This python project of detecting fake news deals with fake and real news.
Using sklearn, we build a TfidfVectorizer on our dataset. Then, we initialize a PassiveAggressive Classifier and fit the model.
In the end, the accuracy score and the confusion matrix tell us how well our model fares.

# The Dataset
The dataset we’ll use for this python project- we’ll call it news.csv. This dataset has a shape of 7796×4.
The first column identifies the news, the second and third are the title and text, and the fourth column has labels denoting whether the news is REAL or FAKE.
The dataset takes up 29.2MB of space and can be downloaded from https://drive.google.com/file/d/1er9NJTLUA3qnRuyhfzuN0XUsoIC4a-_q/view 

# Credit
https://data-flair.training/blogs/advanced-python-project-detecting-fake-news/ - This project would not have been completed without the help and complete guidance from the website.
Also like to thanks SDS BIT Mesra for encouraging me to start with a project to get a hands-on experience of what I'm learning.
I was a complete beginner when I made this project and learned a lot about how to implement a TfidfVectorizer, initialize a PassiveAggressiveClassifier, and fit our model.
