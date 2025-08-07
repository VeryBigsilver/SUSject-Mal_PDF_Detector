from flask import Flask, render_template, request, jsonify
import os
import pickle
import subprocess
import tempfile
import re
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 제한

# 업로드 폴더 생성
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 허용된 파일 확장자
ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_pdf_features(pdf_path):
    """pdfid를 사용하여 PDF 특징 추출"""
    try:
        # pdfid 명령어 실행
        result = subprocess.run(['python', 'pdfid/pdfid.py', pdf_path], 
                              capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        return f"특징 추출 오류: {str(e)}"

def extract_feature_from_pdf(pdf_path):
    """사용자가 제공한 방식으로 PDF 특징 추출"""
    features = {}
    
    # 기본 특징들 초기화
    pdfid_feature_columns = [
        'obj', 'endobj', 'stream', 'endstream', 'xref', 'trailer', 'startxref',
        '/Page', '/Encrypt', '/ObjStm', '/JS', '/JavaScript', '/AA', '/OpenAction',
        '/AcroForm', '/JBIG2Decode', '/RichMedia', '/Launch', '/EmbeddedFile',
        '/XFA', '/URI', '/Colors > 2^24'
    ]
    
    # 모든 특징을 0으로 초기화
    for col in pdfid_feature_columns:
        features[col] = 0
    
    # 파일 크기 추가
    try:
        features['size'] = os.path.getsize(pdf_path)
    except:
        features['error'] = True
        return features
    
    # pdfid로 정보 추출
    try:
        result = subprocess.run(['python', 'pdfid/pdfid.py', pdf_path], 
                              capture_output=True, text=True, check=True)
        
        # pdfid 출력 파싱
        extracted_features = False
        for line in result.stdout.strip().split('\n'):
            if line.startswith(' '):
                parts = line.strip().split()
                if len(parts) >= 2 and parts[-1].isdigit():
                    feature_name = ' '.join(parts[:-1])
                    feature_count = int(parts[-1])
                    features[feature_name] = feature_count
                    extracted_features = True
        
        # 아무 특징도 추출되지 않으면 손상된 PDF로 판단
        if not extracted_features:
            features['corrupted'] = True
            features['error'] = True
            return features
        
        features['is_pdf'] = True
        
        # diff 정보 계산
        features['obj_diff'] = abs(features.get('obj', 0) - features.get('endobj', 0))
        features['stream_diff'] = abs(features.get('stream', 0) - features.get('endstream', 0))
        features['xref_diff'] = abs(features.get('xref', 0) - features.get('startxref', 0))
        
    except subprocess.CalledProcessError as e:
        features['error'] = True
        features['corrupted'] = True
    except Exception as e:
        features['error'] = True
        features['corrupted'] = True
    
    return features

def create_model_feature_vector(features_dict):
    """모델 학습에 사용된 특징들만 추출하여 벡터 생성"""
    # 모델 학습에 사용되지 않은 특징들 제외
    excluded_features = ['obj', 'endobj', 'stream', 'endstream', 'xref', 'startxref', 'trailer']
    
    # 모델 학습에 사용된 특징들 (추정)
    model_features = [
        'size', '/Page', '/Encrypt', '/ObjStm', '/JS', '/JavaScript', 
        '/AA', '/OpenAction', '/AcroForm', '/JBIG2Decode', '/RichMedia', 
        '/Launch', '/EmbeddedFile', '/XFA', '/URI', '/Colors > 2^24',
        'obj_diff', 'stream_diff', 'xref_diff'
    ]
    
    feature_vector = []
    for feature in model_features:
        feature_vector.append(features_dict.get(feature, 0))
    
    return np.array(feature_vector).reshape(1, -1)

def load_model():
    """저장된 모델 로드"""
    try:
        model_path = 'model/model_randomforest.pkl'
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"모델 로드 오류: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # 1. PDF 특징 추출
            features_dict = extract_feature_from_pdf(filepath)
            
            if features_dict.get('error', False):
                # 손상된 파일도 분석창으로 넘어가서 표시하도록 수정
                if features_dict.get('corrupted', False):
                    # 손상된 파일 정보를 포함한 결과 반환
                    return jsonify({
                        'success': True,
                        'filename': filename,
                        'prediction': '손상된 PDF 파일',
                        'confidence': '100%',
                        'analysis_method': '파일 구조 분석',
                        'feature_counts': features_dict,
                        'is_corrupted': True,
                        'corruption_message': '파일이 올바른 PDF 형식이 아니거나 손상되었습니다.'
                    })
                else:
                    return jsonify({'error': 'PDF 파일 분석 중 오류가 발생했습니다.'}), 500
            
            # 2. 원본 pdfid 출력 (웹 표시용)
            pdfid_output = extract_pdf_features(filepath)
            
            # 3. AI 모델 로드 및 예측
            model = load_model()
            if not model:
                os.remove(filepath)
                return jsonify({'error': 'AI 모델을 로드할 수 없습니다.'}), 500
            
            # 모델 예측 수행
            feature_vector = create_model_feature_vector(features_dict)
            prediction_result = model.predict(feature_vector)[0]
            prediction_proba = model.predict_proba(feature_vector)[0]
            
            if prediction_result == 1:
                result = "악성 PDF (AI 모델 판정)"
                confidence = f"{prediction_proba[1]*100:.1f}%"
            else:
                result = "정상 PDF (AI 모델 판정)"
                confidence = f"{prediction_proba[0]*100:.1f}%"
            
            # 임시 파일 삭제
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'filename': filename,
                'features': pdfid_output,  # 원본 pdfid 출력
                'prediction': result,
                'confidence': confidence,
                'analysis_method': "AI 모델",
                'feature_counts': features_dict  # 추출된 특징들
            })
            
        except Exception as e:
            # 오류 발생 시 파일 삭제
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'분석 중 오류가 발생했습니다: {str(e)}'}), 500
    
    return jsonify({'error': '허용되지 않는 파일 형식입니다.'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 