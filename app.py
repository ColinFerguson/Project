from collections import Counter
from flask import Flask, request, render_template
import cPickle as pickle
import pandas as pd
from test import *


with open('tfidf_model_and_fitted_matrix.pkl', 'r') as f:
    prediction_model, prediction_matrix = pickle.load(f)

with open('ovr.pkl', 'r') as g:
    ovr = pickle.load(g)

with open('NT_sim_model_and_fitted_matrix.pkl', 'r') as f:
    similarity_model, similarity_matrix = pickle.load(f)

with open('NT_titles_and_urls.pkl', 'r') as f:
    title_list, urls = pickle.load(f)

names = {'AG': 'Algebraic Geometry', 'AT': 'Algebraic Topology',\
         'AP': 'Analysis of PDEs', 'CT': 'Category Theory',\
         'CA': 'Classical Analysis and ODEs', 'CO': 'Combinatorics',\
         'AC': 'Commutative Algebra', 'CV': 'Complex Variables',\
         'DG': 'Differential Geometry', 'DS': 'Dynamical Systems',\
         'FA': 'Functional Analysis', 'GM': 'General Mathematics',\
         'GN': 'General Topology', 'GT': 'Geometric Topology',\
         'GR': 'Group Theory', 'HO': 'History and Overview',\
         'IT': 'Information Theory', 'KT': 'K-Theory and Homology',\
         'LO': 'Logic', 'MP': 'Mathematical Physics',\
         'MG': 'Metric Geometry', 'NT': 'Number Theory',\
         'NA': 'Numerical Analysis', 'OA': 'Operator Algebras',\
         'OC': 'Optimization and Control', 'PR': 'Probability',\
         'QA': 'Quantum Algebra', 'RT': 'Representation Theory',\
         'RA': 'Rings and Algebras', 'SP': 'Spectral Theory',\
         'ST': 'Statistics Theory', 'SG': 'Symplectic Geometry'}

app = Flask(__name__)


# Form page to submit text

@app.route('/')
def index():
    return render_template('homepage_template.html', title='Bikes are fun!')

@app.route('/more/')
def more():
    return render_template('starter_template.html')



@app.route('/submission_page')
def submission_page():
    return '''
        <form action="/prediction_model" method='POST' >
            <input type="text" name="user_input" />
            <input type="submit" />
        </form>
        '''

@app.route('/find_similar')
def find_similar():
    return '''
        <form action="/similarity_model" method='POST' >
            <input type="text" name="user_input" />
            <input type="submit" />
        </form>
        '''
# My prediction app
@app.route('/prediction_model', methods=['POST'] )
def prediction():
    url = (str(request.form['user_input']))

    text = [parsePDF(url)]

    text = clean_pdf_text(text)

    vector = prediction_model.transform(text)

    pred = ovr.predict(vector)

    return pred[0]

#My similarity app
@app.route('/similarity_model', methods=['POST'] )
def similarity():

    N = similarity_matrix.todense()

    url = (str(request.form['user_input']))

    user_text = [parsePDF(url)]
    user_text = clean_pdf_text(user_text)
    user_vec = similarity_model.transform(user_text)
    user_dense = user_vec.todense()

    distances = [cosine(x, user_dense) for x in N]
    best_score_indices = np.argsort(distances)[:6]
    best_scores = [np.around(distances[i],3) for i in best_score_indices]
    best_titles = [title_list[i] for i in best_score_indices]
    best_urls = [urls[i] for i in best_score_indices]

    #return 'The closest papers in the past year to yours are: {0}, {1}, {2}'.format(best_titles[1], best_scores[1], best_urls[1])
    # print '\n'
    # for x, y, z in zip(best_titles, best_scores, best_urls):
    #     print x
    #     print 'Score: ', y
    #     print 'URL: ', z
    #     print '\n'
    #
    return best_titles[0]

@app.route('/heatmap')
def heatmap():
    return render_template('heatmap_template.html')

@app.route('/roc')
def roc():
    return render_template('roc_template.html')

@app.route('/category-sim')
def sim():
    return render_template('category-sim_template.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
