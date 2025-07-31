# 악성 PDF 분석 웹 서비스

AI를 이용한 악성 PDF 분석 웹 서비스의 프로토타입입니다.

## 기능

- PDF 파일 업로드 (드래그 앤 드롭 지원)
- pdfid를 사용한 PDF 특징 추출
- AI 모델을 통한 악성 PDF 분석

## 설치 및 실행

### 1. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

### 2. 웹 서비스 실행
```bash
python app.py
```

### 3. 브라우저에서 접속
```
http://localhost:5000
```

## 프로젝트 구조

```
susject/
├── app.py              # Flask 메인 애플리케이션
├── templates/
│   └── index.html      # 웹 인터페이스
├── model/              # AI 모델 파일들 (pkl 형식)
├── pdfid/              # PDF 특징 추출 도구
├── uploads/            # 임시 업로드 폴더
├── requirements.txt    # Python 패키지 목록
└── README.md          # 이 파일
```

## 사용법

1. 웹 브라우저에서 `http://localhost:5000`에 접속
2. PDF 파일을 드래그하거나 "파일 선택" 버튼을 클릭하여 업로드
3. 자동으로 PDF 특징이 추출되고 AI 모델이 분석을 수행
4. 분석 결과를 확인

## 주의사항

- 현재는 기본적인 구조만 구현되어 있습니다
- `model/malicious_pdf_model.pkl` 파일이 필요합니다
- `pdfid` 도구가 정상적으로 작동해야 합니다
- 업로드된 파일은 분석 후 자동으로 삭제됩니다

## 향후 개선 사항

- 실제 AI 모델 연동
- 더 정확한 특징 추출 및 전처리
- 분석 결과 시각화
- 보안 강화 