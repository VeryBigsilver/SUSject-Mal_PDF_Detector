from flask import Flask, render_template, request, jsonify
import os
import subprocess
import tempfile
import re
from werkzeug.utils import secure_filename
from analyze import analyze_pdf_with_ai, extract_pdf_features

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB 제한

# PDF 분석 도구 경로
PDF_PARSER_PATH = 'pdf-parser.py'
PDFDATA_PATH = 'uploads'

# 업로드 폴더 생성
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 허용된 파일 확장자
ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def search_mz_pk_in_pdf(filename, pdf_parser_path=PDF_PARSER_PATH, pdfdata_path=PDFDATA_PATH):
    """pdf-parser.py를 사용하여 PDF에서 MZ와 PK 문자열 검색"""
    pdf_file_path = os.path.join(pdfdata_path, filename)

    if not os.path.exists(pdf_file_path):
        error_message = f"Error: File not found at {pdf_file_path}"
        return error_message, error_message # Return error for both searches

    mz_output = None
    pk_output = None

    try:
        # Search for "MZ"
        result_mz = subprocess.run(
            ['python', pdf_parser_path, pdf_file_path, '--search', 'MZ'],
            capture_output=True,
            check=True,
            text=True,
            timeout=60
        )
        mz_output = result_mz.stdout

    except subprocess.CalledProcessError as e:
        mz_output = f"Error executing pdf-parser.py for {filename} searching for 'MZ': {e.stderr}"
    except FileNotFoundError:
        mz_output = f"Error: pdf-parser.py not found at {pdf_parser_path}"
    except subprocess.TimeoutExpired:
        mz_output = f"Error: pdf-parser.py timed out for {filename} searching for 'MZ'"
    except Exception as e:
        mz_output = f"An unexpected error occurred for {filename} searching for 'MZ': {e}"

    try:
        # Search for "PK"
        result_pk = subprocess.run(
            ['python', pdf_parser_path, pdf_file_path, '--search', 'PK'],
            capture_output=True,
            check=True,
            text=True,
            timeout=60
        )
        pk_output = result_pk.stdout

    except subprocess.CalledProcessError as e:
        pk_output = f"Error executing pdf-parser.py for {filename} searching for 'PK': {e.stderr}"
    except FileNotFoundError:
        # Avoid repeating the error message if the file was not found in the first place
        if "File not found" not in mz_output:
             pk_output = f"Error: pdf-parser.py not found at {pdf_parser_path}"
    except subprocess.TimeoutExpired:
         pk_output = f"Error: pdf-parser.py timed out for {filename} searching for 'PK'"
    except Exception as e:
        pk_output = f"An unexpected error occurred for {filename} searching for 'PK': {e}"

    return mz_output, pk_output


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
            # 1. 모델 타입 가져오기 (기본값: randomforest)
            model_type = request.form.get('model_type', 'randomforest')
            
            # 2. MZ/PK 문자열 검색 (pdf-parser.py 사용)
            mz_output, pk_output = search_mz_pk_in_pdf(filename)
            
            # 3. 원본 pdfid 출력 (웹 표시용)
            pdfid_output = extract_pdf_features(filepath)
            
            # 4. AI 분석 수행
            ai_result = analyze_pdf_with_ai(filepath, mz_output, pk_output, model_type)
            
            if 'error' in ai_result:
                os.remove(filepath)
                return jsonify(ai_result), 500
            
            # 임시 파일 삭제
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'filename': original_filename,
                'features': pdfid_output,  # 원본 pdfid 출력
                'prediction': ai_result['prediction'],
                'confidence': ai_result['confidence'],
                'analysis_method': ai_result['analysis_method'],
                'feature_counts': ai_result['feature_counts'],
                'is_corrupted': ai_result.get('is_corrupted', False),
                'corruption_message': ai_result.get('corruption_message', ''),
                'model_type': ai_result.get('model_type', model_type)
            })
            
        except Exception as e:
            # 오류 발생 시 파일 삭제
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'분석 중 오류가 발생했습니다: {str(e)}'}), 500
    
    return jsonify({'error': '허용되지 않는 파일 형식입니다.'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 