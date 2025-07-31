import logging
import numpy as np
from typing import Optional, List

def preprocess_pdf(filepath: str) -> Optional[List[float]]:
    """
    Preprocess PDF file for malware analysis.
    This is a placeholder implementation - replace with your actual preprocessing code.
    
    Args:
        filepath (str): Path to the PDF file
        
    Returns:
        Optional[List[float]]: Feature vector for ML model, or None if processing fails
    """
    try:
        # Import PDF processing libraries
        try:
            import PyPDF2
        except ImportError:
            try:
                import pdfplumber
            except ImportError:
                logging.error("No PDF processing library available. Install PyPDF2 or pdfplumber.")
                return None
        
        logging.info(f"Processing PDF file: {filepath}")
        
        # Basic PDF analysis features
        features = []
        
        # File size feature
        import os
        file_size = os.path.getsize(filepath)
        features.append(file_size)
        
        # PDF structure features using PyPDF2
        try:
            with open(filepath, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Number of pages
                num_pages = len(pdf_reader.pages)
                features.append(num_pages)
                
                # Check for JavaScript (potential indicator of malicious content)
                has_javascript = 0
                text_content = ""
                
                for page in pdf_reader.pages:
                    try:
                        page_text = page.extract_text()
                        text_content += page_text
                        if 'javascript' in page_text.lower() or 'js' in page_text.lower():
                            has_javascript = 1
                    except Exception as e:
                        logging.warning(f"Could not extract text from page: {e}")
                
                features.append(has_javascript)
                
                # Text length
                features.append(len(text_content))
                
                # Number of annotations
                annotations_count = 0
                for page in pdf_reader.pages:
                    if '/Annots' in page:
                        annotations_count += len(page['/Annots'])
                features.append(annotations_count)
                
                # Check for embedded files
                has_embedded_files = 0
                if hasattr(pdf_reader, 'attachments') and pdf_reader.attachments:
                    has_embedded_files = 1
                features.append(has_embedded_files)
                
                # Check for forms
                has_forms = 0
                if '/AcroForm' in pdf_reader.trailer.get('/Root', {}):
                    has_forms = 1
                features.append(has_forms)
                
        except Exception as e:
            logging.error(f"Error processing PDF with PyPDF2: {e}")
            # Fallback to basic features
            features = [file_size, 1, 0, 0, 0, 0, 0]  # Default feature vector
        
        # Ensure we have a consistent feature vector length
        # Pad or truncate to expected size (e.g., 10 features)
        expected_length = 10
        while len(features) < expected_length:
            features.append(0.0)
        features = features[:expected_length]
        
        # Normalize features (basic min-max scaling)
        normalized_features = []
        for i, feature in enumerate(features):
            if i == 0:  # File size - normalize by MB
                normalized_features.append(feature / (1024 * 1024))
            elif i == 1:  # Number of pages - normalize by typical max
                normalized_features.append(min(feature / 100.0, 1.0))
            elif i == 3:  # Text length - normalize by typical max
                normalized_features.append(min(feature / 10000.0, 1.0))
            else:  # Binary features or counts - keep as is or normalize
                normalized_features.append(float(feature))
        
        logging.info(f"Extracted {len(normalized_features)} features: {normalized_features}")
        return normalized_features
        
    except Exception as e:
        logging.error(f"Error preprocessing PDF {filepath}: {str(e)}")
        return None

def extract_metadata_features(pdf_reader) -> List[float]:
    """Extract metadata-based features from PDF"""
    features = []
    
    try:
        metadata = pdf_reader.metadata
        if metadata:
            # Check for suspicious metadata
            suspicious_keywords = ['virus', 'malware', 'exploit', 'shell', 'payload']
            suspicious_count = 0
            
            for key, value in metadata.items():
                if value and isinstance(value, str):
                    for keyword in suspicious_keywords:
                        if keyword.lower() in value.lower():
                            suspicious_count += 1
            
            features.append(suspicious_count)
        else:
            features.append(0)
            
    except Exception as e:
        logging.warning(f"Could not extract metadata: {e}")
        features.append(0)
    
    return features
