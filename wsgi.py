"""WSGI entry point: `waitress-serve --listen=127.0.0.1:5000 wsgi:app` (see start_server.bat)."""
from app import create_app

app = create_app()
