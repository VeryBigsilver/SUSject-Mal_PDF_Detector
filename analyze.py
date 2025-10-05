import pickle
import json
import numpy as np
from typing import Dict, Tuple, Optional, List, Any

# Torch / GNN imports (for GNN model loading)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader

# 전처리(graph 생성) 유틸
from preprocess import build_graph_from_pdf

MODEL_PATH_RF = 'model/model_randomforest.pkl'


def create_model_feature_vector_rf(features_dict: Dict[str, int]) -> np.ndarray:
    """모델 학습에 사용된 특징들만 추출하여 벡터 생성"""
    model_features = [
        'size', '/Page', '/Encrypt', '/ObjStm', '/JS', '/JavaScript',
        '/AA', '/OpenAction', '/AcroForm', '/JBIG2Decode', '/RichMedia',
        '/Launch', '/EmbeddedFile', '/XFA', '/URI', '/Colors > 2^24',
        'has_MZ', 'has_PK', 'obj_diff', 'stream_diff', 'xref_diff'
    ]

    feature_vector = [features_dict.get(feature, 0) for feature in model_features]
    return np.array(feature_vector).reshape(1, -1)


def load_model_rf(model_path: str = MODEL_PATH_RF):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"모델 로드 오류: {e}")
        return None


def predict_with_model_rf(model, feature_vector: np.ndarray) -> Tuple[int, np.ndarray]:
    """모델로 예측 결과와 확률 벡터 반환"""
    prediction_result = model.predict(feature_vector)[0]
    prediction_proba = model.predict_proba(feature_vector)[0]
    return prediction_result, prediction_proba


# ==========================
# GNN 모델 로딩/유틸
# ==========================

class OneClassPDFGNN(nn.Module):
    """One-Class GNN (노드/엣지 복원 포함) 아키텍처.

    - Encoder: GCNConv → GCNConv (hidden 차원 유지)
    - Node Decoder: Linear(hidden→hidden) → Linear(hidden→in_channels)
    - Edge Decoder: 노드 임베딩 내적 (BCEWithLogitsLoss에 사용 가능)

    주의: 저장이 state_dict인 경우 in_channels/hidden이 필요하며, 본 모듈은
    저장된 가중치의 shape로부터 자동 추론을 시도합니다.
    """

    def __init__(self, in_channels: int, hidden: int = 64) -> None:
        super().__init__()
        # Encoder
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        # Node-level decoder
        self.decoder_node_lin1 = nn.Linear(hidden, hidden)
        self.decoder_node_lin2 = nn.Linear(hidden, in_channels)

    def forward(self, data):  # type: ignore[override]
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Encoder
        x = F.relu(self.conv1(x, edge_index))
        node_representation = F.relu(self.conv2(x, edge_index))

        # Node reconstruction
        reconstructed_x = F.relu(self.decoder_node_lin1(node_representation))
        reconstructed_x = self.decoder_node_lin2(reconstructed_x)

        # Edge scores via inner product for existing edges
        row, col = edge_index
        edge_scores = torch.sum(
            node_representation[row] * node_representation[col], dim=1
        )

        return reconstructed_x, edge_scores, node_representation


def load_model_gnn(
    model_path: str = 'model/model_gnn.pt',
    in_channels: Optional[int] = None,
    hidden: Optional[int] = None,
):
    """저장된 GNN 모델을 로드해 반환합니다.

    동작:
    1) 전체 모델 객체가 저장된 경우 그대로 반환
    2) state_dict가 저장된 경우: in_channels가 필요하며 동일 아키텍처를 생성해 가중치 로드

    반환: nn.Module 또는 None (로드 실패 시)
    """
    if not model_path:
        return None

    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    except Exception:
        return None

    # 케이스 1: 전체 모델 객체가 직렬화되어 있는 경우
    if isinstance(checkpoint, nn.Module):
        checkpoint.eval()
        return checkpoint

    # 케이스 2: state_dict 형태인 경우
    state_dict = None
    meta_in_channels: Optional[int] = None

    if isinstance(checkpoint, dict):
        # 흔한 패턴: { 'model_state_dict': ..., 'in_channels': ... }
        if 'model_state_dict' in checkpoint and isinstance(checkpoint['model_state_dict'], dict):
            state_dict = checkpoint['model_state_dict']
            meta_in_channels = checkpoint.get('in_channels', None)
            if isinstance(meta_in_channels, (int,)) and meta_in_channels > 0:
                in_channels = meta_in_channels
        # 순수 state_dict일 수도 있음
        elif all(isinstance(k, str) for k in checkpoint.keys()):
            state_dict = checkpoint

    if state_dict is None:
        return None

    # state_dict에서 in_channels/hidden 자동 추론 시도
    # decoder_node_lin2.weight: (out=in_channels, in=hidden)
    try:
        if 'decoder_node_lin2.weight' in state_dict:
            w = state_dict['decoder_node_lin2.weight']
            if hasattr(w, 'shape') and len(w.shape) == 2:
                inferred_in_channels = int(w.shape[0])
                inferred_hidden = int(w.shape[1])
                if in_channels is None:
                    in_channels = inferred_in_channels
                if hidden is None:
                    hidden = inferred_hidden
    except Exception:
        pass

    # 기본 hidden 설정(미추론 시 64)
    if hidden is None:
        hidden = 64

    if in_channels is None or not isinstance(in_channels, int) or in_channels <= 0:
        # in_channels를 알 수 없으면 동일 아키텍처를 구성할 수 없음
        return None

    try:
        model = OneClassPDFGNN(in_channels=in_channels, hidden=int(hidden))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        # strict=False로 유연하게 로딩; 호출부에서 검증하도록 None 미반환
        model.eval()
        return model
    except Exception:
        return None


def create_model_feature_vector_gnn(features_dict: Dict[str, int]) -> np.ndarray:
    """GNN 입력 전처리 (프로젝트 요구에 맞게 향후 구현)."""
    raise NotImplementedError("GNN 전처리는 프로젝트 설계에 맞춰 별도 구현이 필요합니다.")


def predict_with_gnn(model, gnn_input) -> Tuple[int, np.ndarray]:
    """Deprecated: 사용하지 않음 (아래 run_gnn_on_graph 사용)."""
    raise NotImplementedError("predict_with_gnn는 사용하지 않습니다. run_gnn_on_graph를 사용하세요.")


def run_gnn_on_graph(
    model: nn.Module,
    graph_data,
    device: Optional[str] = None,
) -> float:
    """단일 그래프에 대해 노드 재구성 오차(MSE)를 계산해 반환합니다.

    반환값: reconstruction_error (float)
    """
    if model is None:
        raise ValueError("model이 None 입니다.")

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)
    loader = DataLoader([graph_data], batch_size=1)

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            # 새 모델 출력: reconstructed_x, edge_scores, node_representation
            reconstructed_x, edge_scores, _ = model(batch)
            # 입력 노드 특징과 재구성 특징 간의 MSE (주요 지표)
            mse = torch.nn.functional.mse_loss(reconstructed_x, batch.x).item()

            # 참고: 엣지 BCE 로스(기존 엣지는 1로 가정). 필요 시 확장 가능
            # if edge_scores.numel() > 0:
            #     edge_target = torch.ones_like(edge_scores)
            #     edge_bce = torch.nn.functional.binary_cross_entropy_with_logits(edge_scores, edge_target).item()
            # else:
            #     edge_bce = 0.0

            return mse

    return float('nan')


def run_gnn_on_pdf(
    pdf_path: str,
    all_possible_types: List[str],
    model: nn.Module,
    threshold: Optional[float] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """PDF 경로를 받아 그래프를 생성하고 GNN 재구성 오차로 판단합니다.

    - threshold가 주어지면: 오차 > threshold 를 악성(1)으로 판단
    - 반환 dict:
      {
        'reconstruction_error': float,
        'prediction': Optional[int],  # threshold 제공 시 0(정상)/1(악성)
      }
    """
    graph = build_graph_from_pdf(pdf_path, all_possible_types)
    recon_error = run_gnn_on_graph(model, graph, device=device)

    result: Dict[str, Any] = {
        'reconstruction_error': recon_error,
    }

    if threshold is not None:
        result['prediction'] = 1 if recon_error > threshold else 0

    return result


# ==========================
# RF + GNN 결합 유틸
# ==========================

def load_all_possible_types(json_path: str = 'model/all_possible_types.json') -> Optional[List[str]]:
    """학습 시 사용한 타입 리스트를 JSON에서 로드합니다."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return data
    except Exception:
        return None
    return None


def combine_rf_gnn(
    rf_prob_malicious: float,
    gnn_reconstruction_error: float,
    threshold_rf: float = 0.6,
    threshold_gnn: float = 0.001026,
) -> Dict[str, Any]:
    """RF 확률과 GNN 재구성 오차를 기반으로 결합 판단을 수행합니다.

    규칙:
      - rf_prob <= threshold_rf OR gnn_error <= threshold_gnn → 악성(1)
      - 그 외 → 정상(0)
    반환:
      {
        'prediction': int,  # 0/1
        'rf_prob': float,
        'gnn_error': float,
        'threshold_rf': float,
        'threshold_gnn': float,
      }
    """
    is_mal = int((rf_prob_malicious >= threshold_rf) or (gnn_reconstruction_error >= threshold_gnn))
    return {
        'prediction': is_mal,
        'rf_prob': float(rf_prob_malicious),
        'gnn_error': float(gnn_reconstruction_error),
        'threshold_rf': float(threshold_rf),
        'threshold_gnn': float(threshold_gnn),
    }


