import pickle
import numpy as np
from typing import Dict, Tuple, Optional


MODEL_PATH = 'model/model_randomforest.pkl'


def create_model_feature_vector(features_dict: Dict[str, int]) -> np.ndarray:
    """모델 학습에 사용된 특징들만 추출하여 벡터 생성"""
    model_features = [
        'size', '/Page', '/Encrypt', '/ObjStm', '/JS', '/JavaScript',
        '/AA', '/OpenAction', '/AcroForm', '/JBIG2Decode', '/RichMedia',
        '/Launch', '/EmbeddedFile', '/XFA', '/URI', '/Colors > 2^24',
        'has_MZ', 'has_PK', 'obj_diff', 'stream_diff', 'xref_diff'
    ]

    feature_vector = [features_dict.get(feature, 0) for feature in model_features]
    return np.array(feature_vector).reshape(1, -1)


def load_model(model_path: str = MODEL_PATH):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"모델 로드 오류: {e}")
        return None


def predict_with_model(model, feature_vector: np.ndarray) -> Tuple[int, np.ndarray]:
    """모델로 예측 결과와 확률 벡터 반환"""
    prediction_result = model.predict(feature_vector)[0]
    prediction_proba = model.predict_proba(feature_vector)[0]
    return prediction_result, prediction_proba


