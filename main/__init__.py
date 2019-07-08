from flask import Flask


def create_app():
    """initializes the app."""
    app = Flask(__name__)

    from .views import main
    app.register_blueprint(main)

    return app