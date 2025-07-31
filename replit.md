# PDF Malware Analyzer

## Overview

This is a Flask-based web application designed to analyze PDF files for potential malware using machine learning. The application provides a simple web interface where users can upload PDF files and receive analysis results indicating whether the file is potentially malicious or safe.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Web Framework**: Flask with Jinja2 templating
- **UI Framework**: Bootstrap 5 for responsive design
- **Icons**: Feather Icons for consistent iconography
- **Client-side**: Vanilla JavaScript for form validation and user interactions
- **File Upload**: HTML5 file input with drag-and-drop styling

### Backend Architecture
- **Main Application**: Flask web server (`app.py`)
- **Entry Point**: Simple WSGI entry point (`main.py`)
- **PDF Processing**: Custom preprocessing module (`pdf_preprocessor.py`)
- **ML Model**: Pickle-serialized machine learning model loaded at runtime
- **File Handling**: Secure file upload with validation and temporary storage

### Template Structure
- **Base Template**: Common layout with navigation and flash messaging
- **Index Template**: Main upload interface and results display
- **Responsive Design**: Bootstrap-based responsive layout

## Key Components

### 1. Web Application (`app.py`)
- Flask application with file upload handling
- Session management with secure secret key
- File validation (PDF only, 16MB max)
- Model loading and prediction integration
- Flash messaging for user feedback

### 2. PDF Preprocessor (`pdf_preprocessor.py`)
- PDF parsing using PyPDF2 or pdfplumber as fallback
- Feature extraction for ML model:
  - File size analysis
  - Page count
  - JavaScript detection
  - Text content analysis
- Returns feature vector for model prediction

### 3. Frontend Interface
- Clean, security-focused design
- File upload with progress indicators
- Real-time validation feedback
- Results display with confidence scores
- Color-coded results (red for malicious, green for safe)

### 4. Static Assets
- Custom CSS with security theme
- JavaScript for enhanced UX
- Form validation and file type checking
- Progress indicators during upload

## Data Flow

1. **File Upload**: User selects PDF file through web interface
2. **Validation**: Client and server-side validation for file type and size
3. **Preprocessing**: PDF is processed to extract ML features
4. **Analysis**: Feature vector is passed to loaded ML model
5. **Results**: Prediction results are displayed with confidence score
6. **Cleanup**: Temporary files are cleaned up after processing

## External Dependencies

### Python Packages
- **Flask**: Web framework
- **PyPDF2**: Primary PDF processing library
- **pdfplumber**: Fallback PDF processing library
- **numpy**: Numerical operations for ML features
- **pickle**: Model serialization/deserialization

### Frontend Dependencies
- **Bootstrap 5**: CSS framework (CDN)
- **Feather Icons**: Icon library (CDN)

### Optional Dependencies
- PDF processing libraries are handled with graceful fallbacks
- Application continues to function with basic features if optional libraries are missing

## Deployment Strategy

### Environment Configuration
- **SESSION_SECRET**: Environment variable for Flask session security
- **MODEL_PATH**: Configurable path to ML model file (defaults to `model.pkl`)
- **Upload Directory**: Configurable upload folder with automatic creation

### File Management
- Temporary file handling with automatic cleanup
- Secure filename generation to prevent path traversal
- Upload size limits enforced at both client and server level

### Security Considerations
- File type validation (PDF only)
- Secure filename handling
- Upload size restrictions
- Session secret management
- Input sanitization for file uploads

### Production Readiness
- Logging configuration for debugging and monitoring
- Error handling for missing models or failed processing
- Graceful degradation when dependencies are unavailable
- Flash messaging system for user feedback

## Notes for Development

- The application expects a pre-trained ML model in pickle format
- PDF processing relies on external libraries that should be installed
- File uploads are stored temporarily and should be cleaned up regularly
- The application is designed to be stateless for easy scaling
- CSS and JavaScript are organized for maintainability and can be easily customized