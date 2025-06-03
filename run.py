import os
from app import create_app
from flask_cors import CORS
#dotenv
from dotenv import load_dotenv
load_dotenv()

app = create_app()

CORS(app)

# Serve static files from Vite dist folder
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_static(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return app.send_static_file(path)
    else:
        return app.send_static_file('index.html')
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(port=port, debug=(os.environ.get("DEBUG", "False").lower() == "true"))