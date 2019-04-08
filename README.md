# Project

Note: the app (www.arxivexplorer.com) is not currently live.

The Arxiv is an academic pre-print respository hosted by Cornell University (www.arxiv.org). 
The goal of this project is to build a recommender system using the arxiv as the dataset. 


There are two algorithms integrated into the app. 
The first is a classification algorithm that predicts the subject classification 
(Number Theory, Algebraic Geometry, etc) of a given paper based on the text.
The algorithm is currently only trained on mathematics papers. 
In the future I would like to extend it to other subjects. The second algorithm is a 
recommendation algorithm, which is currently trained on Number Theory papers. Given a paper, 
the recommender finds the five most similar number theory papers posted on the arxiv in 2015 
(through August). In the future I would like the train the recommender on more subjects.

For the predictions, the data was the text of each paper, and the targets were the given labels.  The data was first
transformed using tfidf, using one- and two-grams and 10000 top features.  Then classification was done
using Logistic Regression and a One vs. All approach.  This was warranted because there are
32 separate mathematics sub-categories.  I tested several different models through cross-validation before
settling on logistic regression.  The ROC curves for each sub-category can be seen on the app and in the 
Create_ROC_and_Heatmaps notebook.  In the end I have an accuracy of about 54%, which is pretty good in such a large
multi-class problem.

The recommender also uses tfidf, but only 150 features.  10000 dimensional space is too large for a recommender because
no vectors are actually close to one another in a dimension so high.  One- and two-grams were also used here.  
Then the distance between each article is computed (I used cosine similarity, 
but other distance metrics could be explored), and similar papers are recommended.

Both of the tfidf models use a custom set of math-specific stop words.  There is also code for latent feature
extraction via NMF and LDA.  That is an avenue I am very excited to explore more fully, but it is
not currently integrated into the app.  I think that looking
at how the features change in time will be very interesting.  More data will need to be collected to fully explore
that.

All the relevant functions live in math_scraping_and_recommending_functions.py, 
and the data collection pipeline can be found in the Math_data_collection notebook.  Details about the classification
 model are in models.py
