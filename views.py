from flask import Blueprint, render_template, request

# main blueprint
main = Blueprint('main', __name__)

@main.route('/', methods=['GET', 'POST'])
def home_page():
    if request.method == 'GET':
        return render_template('index.html')

    if request.method == 'POST':
        return render_template('story.html')