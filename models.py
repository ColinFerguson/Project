'''Models used in the app'''
from math_scraping_and_recommending_functions import *

#First we have the tfidf model for the text

math_df = pd.read_pickle('general_math_text_titles_urls_label_df.pkl')

text_data = math_df['text'].values
targets = math_df['label'].values

text_data = list(text_data)
text_data = clean_pdf_text(text_data)
text_data = np.array(text_data)

tfidf = TfidfVectorizer(max_features=10000, stop_words=math_stop(), \
                    ngram_range=(1, 2), decode_error='ignore')
M = tfidf.fit_transform(text_data)

'''If we need to save it, use this'''
# with open('tfidf_model_and_fitted_matrix.pkl', 'w') as f:
#     pickle.dump((tfidf, M),f)


#Now we actually need to classify something

ovr = OneVsRestClassifier(LogisticRegression(C=1, penalty='l2'), n_jobs=-1)
ovr.fit(M, targets)

'''Again, may need to save it'''

# with open('ovr.pkl', 'w') as f:
#     pickle.dump(ovr, f)

'''Now we are free to make predictions.  Hooray, what fun!'''
