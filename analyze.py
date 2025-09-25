import os
import pickle
import subprocess
import numpy as np
from sklearn.preprocessing import StandardScaler

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

def create_model_feature_vector(features_dict, model_type='randomforest'):
    """모델 학습에 사용된 특징들만 추출하여 벡터 생성"""
    # 모델 학습에 사용되지 않은 특징들 제외
    excluded_features = ['obj', 'endobj', 'stream', 'endstream', 'xref', 'startxref', 'trailer']
    
    # Random Forest 모델용 특징들
    model_features_randomforest = [
        'size', '/Page', '/Encrypt', '/ObjStm', '/JS', '/JavaScript', 
        '/AA', '/OpenAction', '/AcroForm', '/JBIG2Decode', '/RichMedia', 
        '/Launch', '/EmbeddedFile', '/XFA', '/URI', '/Colors > 2^24',
        'has_MZ', 'has_PK',
        'obj_diff', 'stream_diff', 'xref_diff'
    ]
    
    # Isolation Forest + Random Forest 모델용 특징들 (사용자가 정의할 예정)
    model_features_isolate = [
        'size', '/Page', '/Encrypt', '/ObjStm', '/JS', '/JavaScript', 
        '/AA', '/OpenAction', '/AcroForm', '/JBIG2Decode', '/RichMedia', 
        '/Launch', '/EmbeddedFile', '/XFA', '/URI', '/Colors > 2^24',
        'has_MZ', 'has_PK',
        'obj_diff', 'stream_diff', 'xref_diff'
    ]
    
    # 모델 타입에 따라 다른 특징 사용
    if model_type == 'randomforest':
        model_features = model_features_randomforest
    elif model_type == 'isolate_randomforest':
        model_features = model_features_isolate
    else:
        raise ValueError(f"지원하지 않는 모델 타입: {model_type}")
    
    feature_vector = []
    for feature in model_features:
        feature_vector.append(features_dict.get(feature, 0))
    
    return np.array(feature_vector).reshape(1, -1)

def load_model(model_type='randomforest'):
    """저장된 모델 로드"""
    try:
        if model_type == 'randomforest':
            model_path = 'model/model_randomforest.pkl'
        elif model_type == 'isolate_randomforest':
            model_path = 'model/model_isolate_randomforest.pkl'
        else:
            raise ValueError(f"지원하지 않는 모델 타입: {model_type}")
            
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"모델 로드 오류 ({model_type}): {e}")
        return None

def analyze_pdf_with_ai(filepath, mz_output, pk_output, model_type='randomforest'):
    """AI 모델을 사용하여 PDF 분석"""
    try:
        # 1. PDF 특징 추출
        features_dict = extract_feature_from_pdf(filepath)
        
        if features_dict.get('error', False):
            if features_dict.get('corrupted', False):
                return {
                    'success': True,
                    'prediction': '손상된 PDF 파일',
                    'confidence': '100%',
                    'analysis_method': '파일 구조 분석',
                    'feature_counts': features_dict,
                    'is_corrupted': True,
                    'corruption_message': '파일이 올바른 PDF 형식이 아니거나 손상되었습니다.'
                }
            else:
                return {'error': 'PDF 파일 분석 중 오류가 발생했습니다.'}
        
        # 2. MZ/PK 검색 결과에서 boolean feature 추출
        has_MZ = 1 if mz_output and "MZ" in mz_output and "Error" not in mz_output else 0
        has_PK = 1 if pk_output and "PK" in pk_output and "Error" not in pk_output else 0
        
        # features_dict에 MZ/PK feature 추가
        features_dict['has_MZ'] = has_MZ
        features_dict['has_PK'] = has_PK
        
        # 3. AI 모델 로드 및 예측
        model = load_model(model_type)
        if not model:
            return {'error': f'AI 모델을 로드할 수 없습니다. (모델 타입: {model_type})'}
        
        # 모델 예측 수행
        feature_vector = create_model_feature_vector(features_dict, model_type)
        prediction_result = model.predict(feature_vector)[0]
        
        # 모델 타입에 따른 분석 방법 표시
        model_name = "Random Forest" if model_type == 'randomforest' else "Isolation Forest + Random Forest"
        
        if model_type == 'randomforest':
            # Random Forest는 predict_proba 사용
            prediction_proba = model.predict_proba(feature_vector)[0]
            if prediction_result == 1:
                result = f"악성 PDF ({model_name} 모델 판정)"
                confidence = f"{prediction_proba[1]*100:.1f}%"
            else:
                result = f"정상 PDF ({model_name} 모델 판정)"
                confidence = f"{prediction_proba[0]*100:.1f}%"
        else:
            # Isolation Forest는 decision_function 사용
            decision_score = model.decision_function(feature_vector)[0]
            threshold = 0.1693465499874528  # Isolation Forest threshold
            
            # decision_score가 threshold보다 작으면 이상치(악성)
            if decision_score < threshold:
                result = f"악성 PDF ({model_name} 모델 판정)"
                # threshold와의 거리를 기반으로 신뢰도 계산
                # decision_score가 더 작을수록(더 이상치일수록) 높은 신뢰도
                distance_from_threshold = abs(decision_score - threshold)
                confidence_score = min(100, max(50, (distance_from_threshold * 200) + 50))
                confidence = f"{confidence_score:.1f}%"
            else:
                result = f"정상 PDF ({model_name} 모델 판정)"
                # threshold와의 거리를 기반으로 신뢰도 계산
                # decision_score가 threshold에 가까울수록 높은 신뢰도
                distance_from_threshold = abs(decision_score - threshold)
                confidence_score = min(100, max(50, 100 - (distance_from_threshold * 200)))
                confidence = f"{confidence_score:.1f}%"
        
        return {
            'success': True,
            'prediction': result,
            'confidence': confidence,
            'analysis_method': f"AI 모델 ({model_name})",
            'feature_counts': features_dict,
            'model_type': model_type
        }
        
    except Exception as e:
        return {'error': f'분석 중 오류가 발생했습니다: {str(e)}'}
