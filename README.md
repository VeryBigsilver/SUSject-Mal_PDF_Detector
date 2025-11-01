![License](https://img.shields.io/github/license/VeryBigsilver/SUSject-Mal_PDF_Detector)

<img src="https://www.konkuk.ac.kr/sites/konkuk/images/common/logo-top-color.png" width="250"/>    for 2025 건국대학교 학술공모전 (우수)


# PDF 악성코드 탐지 시스템

AI 기반 PDF 악성코드 탐지 웹 서비스입니다. PDF 파일의 특징을 분석하여 악성코드 포함 여부를 실시간으로 탐지합니다.

random forest + GAE

**all research files locate at research_files branch**

## 🚀 주요 기능

### 📊 **분석 방법**
- **기본 모드**: Random Forest 기반 머신러닝 모델
- **plus 모드**: RF + GAE 기반 AI 분석
- **실시간 분석**: 업로드 즉시 분석 시작

### 🔍 **상세한 특징 분석**
- **PDF 구조 분석**: 객체, 스트림, 참조 테이블 등
- **보안 위험 요소**: 자바스크립트, 외부 실행, 임베디드 파일
- **위험도 분류**: 높은 위험, 중간 위험, 안전으로 분류
- **실제 값 표시**: 각 특징의 구체적인 내용 확인

### 🔒 **보안 및 개인정보 보호**
- **임시 파일 처리**: 분석 후 자동 파일 삭제
- **개인정보 보호**: 서버에 파일 영구 저장 안함
- **안전한 업로드**: 파일 형식 검증 및 크기 제한

## 📦 설치 및 실행 (in local)

### 1. 저장소 클론
```bash
git clone https://github.com/VeryBigsilver/SUSject-Mal_PDF_Detector.git
```

### 2. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

### 3. 웹 서비스 실행
```bash
python app.py
```

### 4. 브라우저에서 접속
```
http://localhost:5000
```

## 📁 프로젝트 구조

```
PDFMalwareDetect/
├── app.py                      # Flask 메인 애플리케이션 (라우팅/업로드)
├── analyze.py                  # RF/GNN 로딩 및 추론 유틸리티
├── preprocess.py               # PDF → 그래프/특징 전처리
├── pdf-parser.py               # MZ/PK 검색 스크립트
├── run_production.py           # 프로덕션 실행 스크립트(옵션)
├── gunicorn.conf.py            # Gunicorn 설정(옵션)
├── templates/
│   ├── index.html             # 메인 페이지 (업로드/토글 UI)
│   └── result.html            # 결과 페이지 (분석 결과 표시)
├── model/
│   ├── model_randomforest_weighted.pkl  # Random Forest 모델
│   ├── model_gnn.pt                      # GNN 모델(One-Class, 재구성 기반)
│   └── all_possible_types.json           # 그래프 노드 타입 리스트
├── pdfid/                     # PDF 특징 추출 도구
│   ├── pdfid.py
│   ├── pdfid.ini
│   └── plugin_*.py
├── uploads/                   # 임시 업로드 폴더
├── feature_processed.csv      # 전처리 산출물(옵션)
├── requirements.txt           # Python 패키지 목록
├── LICENSE                    # 라이선스
└── README.md                  # 프로젝트 문서
```

## 🎯 사용법

### 1. 파일 업로드
- 웹 브라우저에서 `http://localhost:5000` 접속
- PDF 파일을 드래그 앤 드롭하거나 "파일 선택" 버튼 클릭
- 지원 형식: PDF 파일만

### 2. 분석 과정
- 업로드 즉시 분석 시작
- 진행률 바로 실시간 모니터링
- PDF 특징 추출 및 AI 모델 분석

### 3. 결과 확인
- **위험도 요약**: 전체 위험도 및 신뢰도
- **파일 정보**: 파일명, 분석 방법, 분석 시간
- **추출된 특징**: 위험도 순으로 정렬된 특징 카드
- **상세 설명**: 각 특징의 의미와 위험도
- **실제 값**: 발견된 특징의 구체적인 내용

# 🚀 구현

## 🔍 분석 툴

thanks for the great tool!

pdfid, pdf-parser

link in: https://blog.didierstevens.com/programs/pdf-tools/

## 🔍 분석 데이터

thanks for the great dataset!

### 📋 **CIC dataset**
Maryam Issakhani, Princy Victor, Ali Tekeoglu, and Arash Habibi Lashkari1, “PDF Malware Detection Based on Stacking Learning”, The International Conference on Information Systems Security and Privacy, February 2022

link in: https://www.unb.ca/cic/datasets/pdfmal-2022.html

### 📋 **PDFREP dataset**
R. Liu, R. Joyce, C. Matuszek and C. Nicholas, "Evaluating Representativeness in PDF Malware Datasets: A Comparative Study and a New Dataset," 2023 IEEE International Conference on Big Data (BigData), Sorrento, Italy, 2023

link in: https://ieee-dataport.org/documents/pdfrep

|        | 정상   | 악성    |
| ------ | ---- | ----- |
| CIC    | 7500 | 0     |
| pdfREP | 0    | 19853 |

전체 샘플 수: 27353

## 🛠️ 기술 스택

### **Backend**
- **Flask**: 웹 프레임워크
- **subprocess**: 외부 도구 실행

### **Frontend**
- **HTML5/CSS3**: 반응형 웹 디자인
- **JavaScript**: 동적 인터페이스
- **SVG**: 애니메이션 아이콘
- **Session Storage**: 데이터 전달

### **AI/ML**
- **Random Forest**: 악성 PDF 분류 모델
- **특징 엔지니어링**: PDF 구조 기반 특징 추출
- **Graph Auto Encoder**: 노드/엣지 재구성 기반 이상 탐지 (PyTorch Geometric)
  - 모델 파일: `model/model_randomforest.pkl`, `model/model_gnn.pt`
  - RF와 결합하여 both 모드에서 최종 판단

| 정확도(accuracy) | 정밀도(precision) | 재현율(recall) | f1-score |
| ------------- | -------------- | ----------- | -------- |
| 0.9927        | 0.9931         | 0.9968      | 0.9950   |

## 🔒 보안 특징

### **파일 처리**
- 임시 저장 후 즉시 삭제
- 파일 형식 검증
- 개인정보 보호

### **분석 과정**
- 서버 측 분석으로 클라이언트 보호
- 오류 발생 시 자동 파일 정리
- 안전한 파일 업로드 처리

## 📈 향후 개선 계획

### **기능 개선**
- [ ] APT 공격에 대한 탐지 강화
- [ ] 추가 PDF 특징 분석

---

**개발자**:   VeryBigsilver
**라이선스**: MIT License  
