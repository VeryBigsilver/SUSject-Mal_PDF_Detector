from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from preprocess import (
    extract_feature_from_pdf,
    extract_pdf_features,
    search_mz_pk_in_pdf,
)
from analyze import (
    load_model,
    create_model_feature_vector,
    predict_with_model,
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB 제한

# 업로드 폴더 생성
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 허용된 파일 확장자
ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

 

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
        # 원본 파일명 보존 (한글 파일명 처리)
        original_filename = file.filename
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
                        'filename': original_filename,
                        'prediction': '손상된 PDF 파일',
                        'confidence': '100%',
                        'analysis_method': '파일 구조 분석',
                        'feature_counts': features_dict,
                        'is_corrupted': True,
                        'corruption_message': '파일이 올바른 PDF 형식이 아니거나 손상되었습니다.'
                    })
                else:
                    return jsonify({'error': 'PDF 파일 분석 중 오류가 발생했습니다.'}), 500
            
            # 2. MZ/PK 문자열 검색 (pdf-parser.py 사용)
            mz_output, pk_output = search_mz_pk_in_pdf(filename)
            
            # MZ/PK 검색 결과에서 boolean feature 추출
            has_MZ = 1 if mz_output and "MZ" in mz_output and "Error" not in mz_output else 0
            has_PK = 1 if pk_output and "PK" in pk_output and "Error" not in pk_output else 0
            
            # features_dict에 MZ/PK feature 추가
            features_dict['has_MZ'] = has_MZ
            features_dict['has_PK'] = has_PK
            
            # 3. 원본 pdfid 출력 (웹 표시용)
            pdfid_output = extract_pdf_features(filepath)
            
            # 4. AI 모델 로드 및 예측
            model = load_model()
            if not model:
                os.remove(filepath)
                return jsonify({'error': 'AI 모델을 로드할 수 없습니다.'}), 500
            
            # 모델 예측 수행
            feature_vector = create_model_feature_vector(features_dict)
            prediction_result, prediction_proba = predict_with_model(model, feature_vector)
            
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
                'filename': original_filename,
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