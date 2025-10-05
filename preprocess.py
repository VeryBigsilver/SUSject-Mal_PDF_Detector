import os
import subprocess
import re
from typing import Dict, Tuple, List, Any
from collections import Counter
import math


PDF_PARSER_PATH = 'pdf-parser.py'
PDFID_PATH = 'pdfid/pdfid.py'
PDFDATA_PATH = 'uploads'


def search_mz_pk_in_pdf_rf(filename: str,
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


def extract_pdf_features_rf(pdf_path: str) -> str:
    """pdfid를 사용하여 PDF 특징 추출 (원본 출력 반환)"""
    try:
        result = subprocess.run(['python', PDFID_PATH, pdf_path], capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        return f"특징 추출 오류: {str(e)}"


def extract_feature_from_pdf_rf(pdf_path: str) -> Dict[str, int]:
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


# ==========================
# GNN용 함수 스텁 (미구현)
# ==========================

def extract_feature_from_pdf_gnn(pdf_path: str) -> Dict[str, int]:
    """GNN 입력 생성을 위한 특징 추출 (미구현)"""
    raise NotImplementedError("GNN 특징 추출은 아직 구현되지 않았습니다.")


def extract_pdf_features_gnn(pdf_path: str) -> str:
    """GNN용 원본 특징 표시/덤프 (미구현)"""
    raise NotImplementedError("GNN 원본 특징 출력은 아직 구현되지 않았습니다.")


def parse_pdf_with_pdfparser(pdf_path: str) -> List[Dict[str, Any]]:
    """
    pdf-parser.py를 사용하여 PDF 파일을 파싱하고 객체 정보를 반환한다.
    각 객체는 다음과 같은 dict:
    {
        "obj_id": (번호, 세대),
        "type": "Page" (예시),
        "subtypes": [],
        "JS": 0/1,
        "OpenAction": 0/1,
        "Launch": 0/1,
        "refs": [(3,0), (5,0)],
        "raw_bytes": b"..."
    }
    """

    try:
        # stdout을 binary 모드로 받기 위해 text=False
        result = subprocess.run(
            ["python", PDF_PARSER_PATH, "-c", pdf_path],
            capture_output=True, text=False, timeout=60
        )
        lines = result.stdout.splitlines()
    except subprocess.TimeoutExpired:
        return []
    except Exception:
        return []

    objects: List[Dict[str, Any]] = []
    current_obj: Dict[str, Any] | None = None
    in_stream = False

    # 정규식 패턴
    obj_start_re = re.compile(rb"^obj\s+(\d+)\s+(\d+)")
    ref_re = re.compile(rb"(\d+)\s+(\d+)\s+R")
    type_re = re.compile(rb"/Type\s*/(\w+)")
    subtype_re = re.compile(rb"/Subtype\s*/(\w+)")

    for line in lines:
        line = line.strip()

        # 객체 시작
        m = obj_start_re.match(line)
        if m:
            if current_obj:
                objects.append(current_obj)
            obj_num, gen_num = int(m.group(1)), int(m.group(2))
            current_obj = {
                "obj_id": (obj_num, gen_num),
                "type": "Unknown",
                "subtypes": [],
                "JS": 0,
                "OpenAction": 0,
                "Launch": 0,
                "refs": [],
                "raw_bytes": b"",
            }
            in_stream = False
            continue

        if not current_obj:
            continue

        # 스트림 시작/끝
        if line == b"stream":
            in_stream = True
            continue
        elif line == b"endstream":
            in_stream = False
            continue

        if in_stream:
            current_obj["raw_bytes"] += line + b"\n"
            continue

        # 속성 파싱
        if b"/JS" in line:
            current_obj["JS"] = 1
        if b"/OpenAction" in line:
            current_obj["OpenAction"] = 1
        if b"/Launch" in line:
            current_obj["Launch"] = 1

        # 타입/서브타입
        t = type_re.search(line)
        if t and current_obj["type"] == "Unknown":
            current_obj["type"] = t.group(1).decode("utf-8", errors="ignore")
        s = subtype_re.search(line)
        if s:
            current_obj["subtypes"].append(
                s.group(1).decode("utf-8", errors="ignore")
            )

        # 참조 파싱
        for rm in ref_re.finditer(line):
            rid, rgen = int(rm.group(1)), int(rm.group(2))
            current_obj["refs"].append((rid, rgen))

    if current_obj:
        objects.append(current_obj)

    return objects


def shannon_entropy(data: bytes) -> float:
    """주어진 바이트열에 대한 Shannon 엔트로피 계산."""
    if not data:
        return 0.0
    counts = Counter(data)
    probs = [c / len(data) for c in counts.values()]
    return -sum(p * math.log2(p) for p in probs)


def objects_to_graph(
    objects: List[Dict[str, Any]],
    all_possible_types: List[str],
    add_self_loop: bool = False,
    bidirectional: bool = True,
):
    """
    파싱된 PDF 객체 목록을 PyTorch Geometric Data 객체로 변환합니다.
    all_possible_types: 전체 데이터셋에서 수집한 고유 타입들의 정렬된 리스트
    """
    # 지연 로딩: torch와 Data는 실제로 그래프를 만들 때만 필요
    import torch
    from torch_geometric.data import Data

    if not objects:
        feature_dim = len(all_possible_types) + 6
        return Data(
            x=torch.empty((0, feature_dim)),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            y=None,
        )

    # 객체 ID ↔ 인덱스 매핑
    id2idx = {str(obj.get('obj_id', i)): i for i, obj in enumerate(objects)}
    idx2id = {i: str(obj.get('obj_id', i)) for i, obj in enumerate(objects)}  # noqa: F841 (참고용)

    # 타입 인코딩 준비
    if 'Unknown' not in all_possible_types:
        all_possible_types = ['Unknown'] + list(all_possible_types)
    type2idx = {t: i for i, t in enumerate(all_possible_types)}

    # 노드 특징 생성
    node_features: List[List[float]] = []
    for obj in objects:
        type_onehot = [0] * len(all_possible_types)
        obj_type = obj.get('type', 'Unknown')
        if obj_type not in type2idx:
            obj_type = 'Unknown'
        type_onehot[type2idx[obj_type]] = 1

        raw_bytes: bytes = obj.get('raw_bytes', b'')
        entropy_feat = shannon_entropy(raw_bytes)
        length_feat = float(len(raw_bytes))

        js_feat = float(int(obj.get('JS', 0)))
        openaction_feat = float(int(obj.get('OpenAction', 0)))
        launch_feat = float(int(obj.get('Launch', 0)))
        depth_feat = float(int(obj.get('depth', 0)))

        feature: List[float] = [
            entropy_feat,
            length_feat,
            js_feat,
            openaction_feat,
            launch_feat,
            depth_feat,
        ] + type_onehot
        node_features.append(feature)

    x = torch.tensor(node_features, dtype=torch.float)

    # 엣지 생성
    edges: List[List[int]] = []
    for i, obj in enumerate(objects):
        for ref_id in obj.get('refs', []):
            ref_id_str = str(ref_id)
            if ref_id_str in id2idx:
                j = id2idx[ref_id_str]
                edges.append([i, j])
                if bidirectional:
                    edges.append([j, i])

    if add_self_loop:
        for i in range(len(objects)):
            edges.append([i, i])

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index)


def build_graph_from_pdf(pdf_path: str, all_possible_types: List[str]):
    """편의 함수: 경로에서 파싱 → 그래프 변환까지 수행."""
    objects = parse_pdf_with_pdfparser(pdf_path)
    return objects_to_graph(objects, all_possible_types)

