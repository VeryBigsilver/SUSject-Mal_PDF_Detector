import os
import subprocess
from typing import Dict, Tuple


PDF_PARSER_PATH = 'pdf-parser.py'
PDFID_PATH = 'pdfid/pdfid.py'
PDFDATA_PATH = 'uploads'


def search_mz_pk_in_pdf(filename: str,
                        pdf_parser_path: str = PDF_PARSER_PATH,
                        pdfdata_path: str = PDFDATA_PATH) -> Tuple[str, str]:
    """pdf-parser.py를 사용하여 PDF에서 MZ와 PK 문자열 검색"""
    pdf_file_path = os.path.join(pdfdata_path, filename)

    if not os.path.exists(pdf_file_path):
        error_message = f"Error: File not found at {pdf_file_path}"
        return error_message, error_message

    mz_output = None
    pk_output = None

    try:
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
        if "File not found" not in (mz_output or ''):
            pk_output = f"Error: pdf-parser.py not found at {pdf_parser_path}"
    except subprocess.TimeoutExpired:
        pk_output = f"Error: pdf-parser.py timed out for {filename} searching for 'PK'"
    except Exception as e:
        pk_output = f"An unexpected error occurred for {filename} searching for 'PK': {e}"

    return mz_output, pk_output


def extract_pdf_features(pdf_path: str) -> str:
    """pdfid를 사용하여 PDF 특징 추출 (원본 출력 반환)"""
    try:
        result = subprocess.run(['python', PDFID_PATH, pdf_path], capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        return f"특징 추출 오류: {str(e)}"


def extract_feature_from_pdf(pdf_path: str) -> Dict[str, int]:
    """pdfid 실행 결과를 파싱해 특징 딕셔너리 추출"""
    features: Dict[str, int] = {}

    pdfid_feature_columns = [
        'obj', 'endobj', 'stream', 'endstream', 'xref', 'trailer', 'startxref',
        '/Page', '/Encrypt', '/ObjStm', '/JS', '/JavaScript', '/AA', '/OpenAction',
        '/AcroForm', '/JBIG2Decode', '/RichMedia', '/Launch', '/EmbeddedFile',
        '/XFA', '/URI', '/Colors > 2^24'
    ]

    for col in pdfid_feature_columns:
        features[col] = 0

    try:
        features['size'] = os.path.getsize(pdf_path)
    except Exception:
        features['error'] = True
        return features

    try:
        result = subprocess.run(['python', PDFID_PATH, pdf_path], capture_output=True, text=True, check=True)
        extracted_features = False
        for line in result.stdout.strip().split('\n'):
            if line.startswith(' '):
                parts = line.strip().split()
                if len(parts) >= 2 and parts[-1].isdigit():
                    feature_name = ' '.join(parts[:-1])
                    feature_count = int(parts[-1])
                    features[feature_name] = feature_count
                    extracted_features = True

        if not extracted_features:
            features['corrupted'] = True
            features['error'] = True
            return features

        features['is_pdf'] = True
        features['obj_diff'] = abs(features.get('obj', 0) - features.get('endobj', 0))
        features['stream_diff'] = abs(features.get('stream', 0) - features.get('endstream', 0))
        features['xref_diff'] = abs(features.get('xref', 0) - features.get('startxref', 0))
    except subprocess.CalledProcessError:
        features['error'] = True
        features['corrupted'] = True
    except Exception:
        features['error'] = True
        features['corrupted'] = True

    return features


