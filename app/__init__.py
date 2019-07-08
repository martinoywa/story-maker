from flask import Flask

"""initializes the app."""
app = Flask(__name__)

from .views import main
app.register_blueprint(main)