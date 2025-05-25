import os
from app import create_app
from flask_cors import CORS
#dotenv
from dotenv import load_dotenv
load_dotenv()

app = create_app()

CORS(app)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(port=port, debug=(os.environ.get("DEBUG", "False").lower() == "true"))