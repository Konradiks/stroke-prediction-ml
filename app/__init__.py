from flask import Flask
from app.routes import main as routes

def create_app():
    app = Flask(__name__, static_folder='dist', static_url_path='')
    app.config['JSON_SORT_KEYS'] = False  # Optional: Prevents sorting of JSON keys
    app.register_blueprint(routes)
    return app

app = create_app()