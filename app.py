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

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    try:
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(url_for('index'))
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('index'))
        
        if not allowed_file(file.filename):
            flash('Only PDF files are allowed', 'error')
            return redirect(url_for('index'))
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            flash('File uploaded successfully!', 'success')
            return render_template('index.html', 
                                 uploaded_file=filename, 
                                 file_path=filepath,
                                 show_analyze=True)
    
    except RequestEntityTooLarge:
        flash('File is too large. Maximum size is 16MB.', 'error')
        return redirect(url_for('index'))
    except Exception as e:
        app.logger.error(f"Upload error: {str(e)}")
        flash('An error occurred during file upload', 'error')
        return redirect(url_for('index'))

@app.route('/analyze', methods=['POST'])
def analyze_file():
    """Analyze the uploaded PDF file"""
    try:
        filepath = request.form.get('filepath')
        filename = request.form.get('filename')
        
        if not filepath or not os.path.exists(filepath):
            flash('File not found', 'error')
            return redirect(url_for('index'))
        
        if model is None:
            flash('AI model not loaded. Please check server configuration.', 'error')
            return redirect(url_for('index'))
        
        # Preprocess the PDF
        app.logger.info(f"Starting analysis of {filename}")
        features = preprocess_pdf(filepath)
        
        if features is None:
            flash('Error processing PDF file', 'error')
            return redirect(url_for('index'))
        
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
        
    except Exception as e:
        app.logger.error(f"Analysis error: {str(e)}")
        flash('An error occurred during analysis', 'error')
        return redirect(url_for('index'))

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
