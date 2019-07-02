from flask import Blueprint

# main blueprint
main = Blueprint('main')

@main.route('/')
def home_page():
    return "<h1>Hello Story Maker</h1>"