
# coding: utf-8

# A Naive Bayes classifier is a probabilistic machine learning model that’s used for classification task.In simple terms, a Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature.
# 
# For example,a dress may be considered to be a shirt if it is red, printed, and has a sleeve. Even if these features depend on each other or upon the existence of the other features, all of these properties independently contribute to the probability that this cloth is a shirt and that is why it is known as ‘Naive’.
# 
# Given a Hypothesis H and evidence E, Bayes’ Theorem states that the relationship between the probability of Hypothesis before getting the evidence P(H) and the probability of the hypothesis after getting the evidence P(H|E) is :
# 
# P(H|E) = P(E|H).P(H)/P(E)
# 
# This relates the probability of the hypothesis before getting the evidence P(H), to the probability of the hypothesis after getting the evidence,
# 
# P(H|E). For this reason,  is called the prior probability, while P(H|E) is called the posterior probability. The factor that relates the two, P(H|E) / P(E), is called the likelihood ratio. Using these terms, Bayes’ theorem can be rephrased as:
# 
#  
# “The posterior probability equals the prior probability times the likelihood ratio.”

# # What is the case of using NB:

# You are working on a classification problem and you have generated your set of hypothesis, created features and discussed the importance of variables. Within an hour, stakeholders want to see the first cut of the model.
# 
# What will you do? You have hunderds of thousands of data points and quite a few variables in your training data set. In such situation, if I were at your place, I would have used ‘Naive Bayes‘, which can be extremely fast relative to other classification algorithms. It works on Bayes theorem of probability to predict the class of unknown data set.

# # let’s see where is it used in the Industry?

# News Categorization, Weather Prediction,Medical Diagnosis, Spam Filtering, Face recognition, Digit recognition...etc
# 

# EXAMPLE:
# Let's set out on a journey by train to create our first very simple Naive Bayes Classifier. Let us assume we are in the city of Hamburg and we want to travel to Munich. We will have to change trains in Frankfurt am Main. We know from previous train journeys that our train from Hamburg might be delayed and the we will not catch our connecting train in Frankfurt. The probability that we will not be in time for our connecting train depends on how high our possible delay will be. The connecting train will not wait for more than five minutes. Sometimes the other train is delayed as well.
# 
# The following lists 'in_time' (the train from Hamburg arrived in time to catch the connecting train to Munich) and 'too_late' (connecting train is missed) are data showing the situation over some weeks. The first component of each tuple shows the minutes the train was late and the second component shows the number of time this occurred.

# In[8]:


# the tuples consist of (delay time of train1, number of times)
# tuples are (minutes, number of times)
in_time = [(0, 22), (1, 19), (2, 17), (3, 18),
           (4, 16), (5, 15), (6, 9), (7, 7),
           (8, 4), (9, 3), (10, 3), (11, 2)]
in_time


# In[9]:


too_late = [(6, 6), (7, 9), (8, 12), (9, 17), 
            (10, 18), (11, 15), (12,16), (13, 7),
            (14, 8), (15, 5)]
too_late


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
X, Y = zip(*in_time)
X2, Y2 = zip(*too_late)
bar_width = 0.9
plt.bar(X, Y, bar_width,  color="blue", alpha=0.75, label="in time")
bar_width = 0.8
plt.bar(X2, Y2, bar_width,  color="red", alpha=0.75, label="too late")
plt.legend(loc='upper right')
plt.show()


# From this data we can deduce that the probability of catching the connecting train if we are one minute late is 1, because we had 19 successful cases experienced and no misses, i.e. there is no tuple with 1 as the first component in 'too_late'.
# 
# We will denote the event "train arrived in time to catch the connecting train" with S (success) and the 'unlucky' event "train arrived too late to catch the connecting train" with M (miss)
# 
# We can now define the probability "catching the train given that we are 1 minute late" formally:
# 
# P(S|1)=19/19=1
# We used the fact that the tuple (1,19) is in 'in_time' and there is no tuple with the first component 1 in 'too_late'
# 
# It's getting critical for catching the connecting train to Munich, if we are 6 minutes late. Yet, the chances are still 60 %:
# 
# P(S|6)=9/9+6=0.6
# Accordingly, the probability for missing the train knowing that we are 6 minutes late is:
# 
# P(M|6)=6/9+6=0.4

# In[10]:


#We can write a 'classifier' function, which will give the probability for catching the connecting train:

in_time_dict = dict(in_time)
too_late_dict = dict(too_late)
def catch_the_train(min):
    s = in_time_dict.get(min, 0)
    if s == 0:
        return 0
    else:
        m = too_late_dict.get(min, 0)
        return s / (s + m)
for minutes in range(-1, 13):
    print(minutes, catch_the_train(minutes))

