import os
from app import create_app
#dotenv
from dotenv import load_dotenv
load_dotenv()

app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(port=port, debug=(os.environ.get("DEBUG", "False").lower() == "true"))