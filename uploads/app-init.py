from flask import Flask
import os

def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__, template_folder='../templates')
    app.secret_key = os.urandom(24)
    
    # Import and register routes
    from app.routes import register_routes
    register_routes(app)
    
    return app
