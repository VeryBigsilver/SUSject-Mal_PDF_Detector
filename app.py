import os
import logging
import pickle
from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import tempfile
import shutil
from pdf_preprocessor import preprocess_pdf

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")

# Configuration
UPLOAD_FOLDER = 'uploads'
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variable to store the loaded model
model = None

def load_model():
    """Load the pre-trained AI model from pickle file"""
    global model
    try:
        model_path = os.environ.get('MODEL_PATH', 'model.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            app.logger.info(f"Model loaded successfully from {model_path}")
        else:
            app.logger.warning(f"Model file not found at {model_path}")
    except Exception as e:
        app.logger.error(f"Error loading model: {str(e)}")

def allowed_file(filename):
    """Check if the uploaded file is a PDF"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main page for file upload"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_file():
    """Analyze the uploaded PDF file"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file selected'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Only PDF files are allowed'}), 400
        
        if model is None:
            return jsonify({'error': 'AI model not loaded. Please check server configuration.'}), 500
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess the PDF
        app.logger.info(f"Starting analysis of {filename}")
        features = preprocess_pdf(filepath)
        
        if features is None:
            # Clean up file
            try:
                os.remove(filepath)
            except Exception:
                pass
            return jsonify({'error': 'Error processing PDF file'}), 400
        
        # Run inference
        prediction = model.predict([features])[0]
        confidence = model.predict_proba([features])[0]
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except Exception as e:
            app.logger.warning(f"Could not remove temporary file: {e}")
        
        # Determine result theme and message
        is_malicious = prediction == 1
        confidence_score = max(confidence) * 100
        
        result_data = {
            'filename': filename,
            'prediction': int(prediction),
            'is_malicious': is_malicious,
            'confidence': round(confidence_score, 2),
            'message': 'MALICIOUS FILE DETECTED' if is_malicious else 'FILE IS CLEAN',
            'theme': 'danger' if is_malicious else 'success'
        }
        
        app.logger.info(f"Analysis complete: {result_data}")
        return render_template('index.html', result=result_data)
        
    except RequestEntityTooLarge:
        return jsonify({'error': 'File is too large. Maximum size is 16MB.'}), 400
    except Exception as e:
        app.logger.error(f"Analysis error: {str(e)}")
        return jsonify({'error': 'An error occurred during analysis'}), 500

@app.errorhandler(413)
def too_large(e):
    flash('File is too large. Maximum size is 16MB.', 'error')
    return redirect(url_for('index'))

@app.errorhandler(404)
def not_found(e):
    return render_template('index.html'), 404

@app.errorhandler(500)
def server_error(e):
    app.logger.error(f"Server error: {str(e)}")
    flash('An internal server error occurred', 'error')
    return render_template('index.html'), 500

# Load model on startup
with app.app_context():
    load_model()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
