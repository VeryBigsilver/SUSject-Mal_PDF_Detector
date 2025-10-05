from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename

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
            # 모드: rf (기본), both (rf + gnn)
            mode = request.form.get('mode', 'rf')

            # 필요한 모듈/함수들을 지연 로딩 (lazy import)
            from preprocess import (
                extract_feature_from_pdf_rf,
                extract_pdf_features_rf,
                search_mz_pk_in_pdf_rf,
            )
            from analyze import (
                load_model_rf,
                create_model_feature_vector_rf,
                predict_with_model_rf,
                load_all_possible_types,
            )

            # 1. PDF 특징 추출 (RF)
            features_dict = extract_feature_from_pdf_rf(filepath)
            
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
            
            # 2. MZ/PK 문자열 검색 (pdf-parser.py 사용, RF)
            mz_output, pk_output = search_mz_pk_in_pdf_rf(filename)
            
            # MZ/PK 검색 결과에서 boolean feature 추출
            has_MZ = 1 if mz_output and "MZ" in mz_output and "Error" not in mz_output else 0
            has_PK = 1 if pk_output and "PK" in pk_output and "Error" not in pk_output else 0
            
            # features_dict에 MZ/PK feature 추가
            features_dict['has_MZ'] = has_MZ
            features_dict['has_PK'] = has_PK
            
            # 3. 원본 pdfid 출력 (웹 표시용, RF)
            pdfid_output = extract_pdf_features_rf(filepath)
            
            # 4. AI 모델 로드 및 예측 (RF)
            model = load_model_rf()
            if not model:
                os.remove(filepath)
                return jsonify({'error': 'AI 모델을 로드할 수 없습니다.'}), 500
            
            # 모델 예측 수행
            feature_vector = create_model_feature_vector_rf(features_dict)
            prediction_result, prediction_proba = predict_with_model_rf(model, feature_vector)
            
            rf_prob_mal = float(prediction_proba[1])

            # 기본 응답 (RF 단독)
            final_prediction_label = "악성 PDF (AI 모델 판정)" if prediction_result == 1 else "정상 PDF (AI 모델 판정)"
            final_confidence = f"{(prediction_proba[1] if prediction_result == 1 else prediction_proba[0])*100:.1f}%"

            response_payload = {
                'success': True,
                'filename': original_filename,
                'features': pdfid_output,
                'prediction': final_prediction_label,
                'confidence': final_confidence,
                'analysis_method': "AI 모델 (RF)",
                'feature_counts': features_dict,
                'rf_probability_malicious': rf_prob_mal,
            }

            # 5. both 모드: RF + GNN 결합 판단
            if mode == 'both':
                all_types = load_all_possible_types()  # 기본 경로: model/all_possible_types.json
                if not all_types:
                    # 타입 리스트가 없으면 GNN을 건너뛰고 경고만 표시
                    response_payload['gnn_notice'] = 'GNN 타입 리스트를 찾을 수 없어 RF만 수행했습니다.'
                    response_payload['analysis_method'] = "AI 모델 (RF)"
                else:
                    # GNN 관련 유틸은 실제 사용할 때만 로드
                    from analyze import load_model_gnn, run_gnn_on_pdf, combine_rf_gnn
                    # in_channels = len(types) + 6 규칙으로 모델 구성 후 로드 시도
                    in_channels = len(all_types) + 6
                    gnn_model = load_model_gnn('model/model_gnn.pt', in_channels=in_channels)
                    if gnn_model is None:
                        response_payload['gnn_notice'] = 'GNN 모델을 로드할 수 없어 RF만 수행했습니다.'
                        response_payload['analysis_method'] = "AI 모델 (RF)"
                    else:
                        gnn_out = run_gnn_on_pdf(filepath, all_types, gnn_model)
                        gnn_error = float(gnn_out.get('reconstruction_error', float('nan')))
                        combo = combine_rf_gnn(rf_prob_mal, gnn_error)

                        # 최종 레이블/정보 갱신
                        response_payload['analysis_method'] = "AI 모델 (RF + GNN)"
                        response_payload['gnn_reconstruction_error'] = gnn_error
                        response_payload['combined_prediction'] = int(combo['prediction'])
                        if combo['prediction'] == 1:
                            response_payload['prediction'] = "악성 PDF (결합 판정)"
                            response_payload['confidence'] = f"RF {rf_prob_mal*100:.1f}% | GNN err {gnn_error:.4f}"
                        else:
                            response_payload['prediction'] = "정상 PDF (결합 판정)"
                            response_payload['confidence'] = f"RF {(1.0-rf_prob_mal)*100:.1f}% | GNN err {gnn_error:.4f}"

            # 임시 파일 삭제 (모든 분석 종료 후)
            os.remove(filepath)

            return jsonify(response_payload)
            
        except Exception as e:
            # 오류 발생 시 파일 삭제
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'분석 중 오류가 발생했습니다: {str(e)}'}), 500
    
    return jsonify({'error': '허용되지 않는 파일 형식입니다.'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 