from collections import Counter
from flask import Flask, request, render_template
import cPickle as pickle
import pandas as pd
from math_scraping_and_recommending_functions import *


with open('tfidf_model_and_fitted_matrix.pkl', 'r') as f:
    prediction_model, prediction_matrix = pickle.load(f)

with open('ovr.pkl', 'r') as g:
    ovr = pickle.load(g)

with open('NT_sim_model_and_fitted_matrix.pkl', 'r') as f:
    similarity_model, similarity_matrix = pickle.load(f)

N = similarity_matrix.todense()

with open('NT_titles_and_urls.pkl', 'r') as f:
    title_list, urls = pickle.load(f)

def three_best(url):
    user_text = [parsePDF(url)]
    user_text = clean_pdf_text(user_text)
    vector = prediction_model.transform(user_text)
    top = np.squeeze(np.argsort(ovr.predict_proba(vector))[:, -3:])
    #top = top[::-1]
    top_probs = np.around(np.squeeze(ovr.predict_proba(vector)[:, top]),3)
    top_probs = top_probs[::-1]
    header = "The most probable classifications for this paper are: "
    class_and_prob = []
    output = ''
    for ix in range(1,4):
        class_and_prob.append(names[ovr.classes_[top][-ix]])

    return render_template('prediction_output_template.html', data=zip(class_and_prob, top_probs))


def NT_sim(url):
    user_text = [parsePDF(url)]
    user_text = clean_pdf_text(user_text)
    user_vec = similarity_model.transform(user_text)
    user_dense = user_vec.todense()

    if np.linalg.norm(user_dense) == 0.0:
        return "Sorry, this paper couldn't be parsed"
    else:
        distances = [cosine(x, user_dense) for x in N]
        best_score_indices = np.argsort(distances)[:6]
        best_scores = [np.around(distances[i], 3) for i in best_score_indices]
        best_titles = [title_list[i][7:] for i in best_score_indices]
        best_urls = [urls[i] for i in best_score_indices]

        return render_template('nt_sim_template.html', data=zip(best_titles, best_scores, best_urls))

names = {'AG': 'Algebraic Geometry', 'AT': 'Algebraic Topology',
         'AP': 'Analysis of PDEs', 'CT': 'Category Theory',
         'CA': 'Classical Analysis and ODEs', 'CO': 'Combinatorics',
         'AC': 'Commutative Algebra', 'CV': 'Complex Variables',
         'DG': 'Differential Geometry', 'DS': 'Dynamical Systems',
         'FA': 'Functional Analysis', 'GM': 'General Mathematics',
         'GN': 'General Topology', 'GT': 'Geometric Topology',
         'GR': 'Group Theory', 'HO': 'History and Overview',
         'IT': 'Information Theory', 'KT': 'K-Theory and Homology',
         'LO': 'Logic', 'MP': 'Mathematical Physics',
         'MG': 'Metric Geometry', 'NT': 'Number Theory',
         'NA': 'Numerical Analysis', 'OA': 'Operator Algebras',
         'OC': 'Optimization and Control', 'PR': 'Probability',
         'QA': 'Quantum Algebra', 'RT': 'Representation Theory',
         'RA': 'Rings and Algebras', 'SP': 'Spectral Theory',
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
    return render_template('predict_category_template.html')


# My prediction app
@app.route('/prediction_model', methods=['POST'])
def prediction():

    url = (str(request.form['user_input']))

    return three_best(url)


# My similarity app
@app.route('/similarity_model', methods=['POST'])
def similarity():

    url = (str(request.form['user_input']))

    return NT_sim(url)


@app.route('/visualizations')
def heatmap():
    return render_template('visualizations_template.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
