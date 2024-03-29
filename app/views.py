from flask import Blueprint, render_template, request, url_for
from .model_files.predict import sample
from .model_files.architecture import load_model

# initialize the model
model = load_model()

# main blueprint
main = Blueprint('main', __name__)

@main.route('/', methods=['GET', 'POST'])
def home_page():
    # home page
    if request.method == 'GET':
        return render_template('index.html')

    # results page
    if request.method == 'POST':
        size = request.form.get('size', type=int)
        prime = request.form.get('prime')
        result = sample(model, size, prime=prime, top_k=5)
        return render_template('story.html', result=result)