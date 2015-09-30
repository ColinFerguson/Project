import cPickle as pickle
from math_scraping_and_recommending_functions import *

user_url = input('Please enter a pdf url: ')

dates = []
for year in ['15', '14', '13', '12']:
    for i in range(12, 0,  -1):
        if len(str(i)) == 1:
            dates.append(year+'0'+str(i))
        else:
            dates.append(year+str(i))
good_dates = dates[3:]  # Just use the first 9 months of 2015,
                        # since the rest of the months haven't happened yet

front = 'http://lanl.arxiv.org/list/math.NT/'
end = '?show=250'
NT_urls = [front+x+end for x in good_dates]

NT_names = ['NT_'+date+'.pkl' for date in good_dates]

NT_tags = [x[:7] for x in NT_names]

for name, tag in zip(NT_names, NT_tags):
    with open(name, 'r') as f:
        tag = pickle.load(f)

text = []
title_list = []
urls = []
for tag in NT_tags:
    for triple in tag:
        text.append(triple[0])
        urls.append(triple[1])
        title_list.append(triple[2])

text = clean_pdf_text(text)

tfidf_NT = TfidfVectorizer(max_features=150, stop_words=math_stop(),
                           ngram_range=(2, 2), decode_error='ignore')
M = tfidf_NT.fit_transform(text)

N = M.todense()

user_text = [parsePDF(user_url)]
user_text = clean_pdf_text(user_text)
user_vec = tfidf_math.transform(user_text)
user_dense = user_vec.todense()

distances = [cosine(x, user_dense) for x in N]
best_score_indices = np.argsort(distances)[:6]
best_scores = [np.around(distances[i], 3) for i in best_score_indices]
best_titles = [title_list[i] for i in best_score_indices]
best_urls = [urls[i] for i in best_score_indices]

print 'The closest papers in the past year to yours are: '
print '\n'
for x, y, z in zip(best_titles, best_scores, best_urls):
    print x
    print 'Score: ', y
    print 'URL: ', z
    print '\n'

# Dists=np.zeros((N.shape[0], N.shape[0]))
# for ix in range(len(Dists)):
#     for jx in range(len(Dists)):
#         Dists[ix, jx]=cosine(N[ix], N[jx])


feature_names = tfidf_NT.get_feature_names()
feature_names = [WordNetLemmatizer().lemmatize(word) for word in feature_names]
nmf = NMF(n_components=10)
nmf.fit(M)
topics = []
for topic_idx, topic in enumerate(nmf.components_):
    topics.append((" ".join([feature_names[i] for
                            i in topic.argsort()[:-5 - 1:-1]])))

print 'The major NMF topics from the corpus are: '
print '\n'
for topic in topics:
    print topic
    print '\n'


# countvec = CountVectorizer(decode_error='ignore', stop_words='english',
#                                max_features=5000, ngram_range=(2,2))
# CV = countvec.fit_transform(text)
# vocab=tuple(countvec.vocabulary_)
# lda_model = lda.LDA(n_topics=10, n_iter=1500)
# lda_model.fit(CV)
#
# topic_word = lda_model.topic_word_  # model.components_ also works
# n_top_words = 8
# topic_words = []
# for i, topic_dist in enumerate(topic_word):
#     topic_words.append(np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1])
#
# print 'The LDA topics are: '
# for i in range(len(topic_words)):
#     print ' '.join(topic_words[i])
#     print '\n'
