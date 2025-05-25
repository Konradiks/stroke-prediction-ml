from flask import Flask
from app.routes import main as routes

def create_app():
    app = Flask(__name__)
    app.config['JSON_SORT_KEYS'] = False  # Optional: Prevents sorting of JSON keys

    # Register routes
    app.register_blueprint(routes)

    return app

app = create_app()