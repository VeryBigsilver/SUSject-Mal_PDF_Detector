import logging
import numpy as np
import pandas as pd
import os
import subprocess
import json
import pickle
import itertools
import hashlib
import math
from typing import Optional, List

def preprocess_pdf(filepath: str) -> Optional[List[float]]:
    """
    Preprocess PDF file for malware analysis using pdfid features.
    
    Args:
        filepath (str): Path to the PDF file
        
    Returns:
        Optional[List[float]]: Feature vector for ML model, or None if processing fails
    """
    try:
        # Initialize feature dataframe structure
        df_feature = {}
        df_feature['is_pdf'] = np.nan
        df_feature['error'] = False
        df_feature['size'] = np.nan
        df_feature['obj_diff'] = np.nan
        df_feature['stream_diff'] = np.nan
        df_feature['xref_diff'] = np.nan

        pdfid_feature_columns = [
            'obj', 'endobj', 'stream', 'endstream', 'xref', 'trailer', 'startxref',
            '/Page', '/Encrypt', '/ObjStm', '/JS', '/JavaScript', '/AA', '/OpenAction',
            '/AcroForm', '/JBIG2Decode', '/RichMedia', '/Launch', '/EmbeddedFile',
            '/XFA', '/URI', '/Colors > 2^24'
        ]
        
        for col in pdfid_feature_columns:
            df_feature[col] = np.nan

        # Extract features using the provided function
        extract_feature(filepath, df_feature)
        
        # Check if there was an error
        if df_feature.get('error', False):
            logging.error(f"Error processing PDF file: {filepath}")
            return None
        
        # Convert to feature vector
        feature_vector = []
        
        # Add basic features
        feature_vector.append(df_feature.get('size', 0))
        feature_vector.append(1 if df_feature.get('is_pdf') else 0)
        feature_vector.append(df_feature.get('obj_diff', 0))
        feature_vector.append(df_feature.get('stream_diff', 0))
        feature_vector.append(df_feature.get('xref_diff', 0))
        
        # Add pdfid features
        for col in pdfid_feature_columns:
            feature_vector.append(df_feature.get(col, 0))
        
        logging.info(f"Extracted {len(feature_vector)} features from {filepath}")
        return feature_vector
        
    except Exception as e:
        logging.error(f"Error preprocessing PDF {filepath}: {str(e)}")
        return None

def extract_feature(pdf_path: str, dataframe: dict):
    """
    Extract features from PDF file using pdfid.
    
    Args:
        pdf_path (str): Path to the PDF file
        dataframe (dict): Dictionary to store features
    """
    # Get PDF file size
    try:
        dataframe['size'] = os.path.getsize(pdf_path)
    except Exception as e:
        logging.error(f"Error getting file size: {e}")
        dataframe['error'] = True
        return

    # Extract features using pdfid
    try:
        # Try to find pdfid in common locations
        pdfid_paths = [
            'pdfid/pdfid.py',  # Downloaded pdfid directory
            'pdfid.py',  # Current directory
            'tools/pdfid.py',  # Tools subdirectory
            '/usr/local/bin/pdfid.py',  # System installation
            'C:/pdfid/pdfid.py'  # Windows installation
        ]
        
        pdfid_path = None
        for path in pdfid_paths:
            if os.path.exists(path):
                pdfid_path = path
                break
        
        if pdfid_path is None:
            logging.error("pdfid.py not found. Please install pdfid or provide correct path.")
            dataframe['error'] = True
            return
        
        # Run pdfid
        result = subprocess.run(['python', pdfid_path, pdf_path], 
                              capture_output=True, text=True, check=True)

        features = {}
        for line in result.stdout.strip().split('\n'):
            if line.startswith(' '):
                parts = line.strip().split()
                if len(parts) >= 2 and parts[-1].isdigit():
                    feature_name = ' '.join(parts[:-1])
                    feature_count = int(parts[-1])
                    features[feature_name] = feature_count
        
        # Store features in dataframe
        for key, val in features.items():
            if key in dataframe:
                dataframe[key] = val
        
        dataframe['is_pdf'] = True

        # Calculate diff features
        if 'obj' in dataframe and 'endobj' in dataframe:
            dataframe['obj_diff'] = abs(dataframe.get('obj', 0) - dataframe.get('endobj', 0))
        
        if 'stream' in dataframe and 'endstream' in dataframe:
            dataframe['stream_diff'] = abs(dataframe.get('stream', 0) - dataframe.get('endstream', 0))
        
        if 'xref' in dataframe and 'startxref' in dataframe:
            dataframe['xref_diff'] = abs(dataframe.get('xref', 0) - dataframe.get('startxref', 0))

    except subprocess.CalledProcessError as e:
        logging.error(f"pdfid subprocess error: {e}")
        dataframe['error'] = True
        return
    except Exception as e:
        logging.error(f"Error in extract_feature: {e}")
        dataframe['error'] = True
        return
