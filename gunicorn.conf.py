# Gunicorn 설정 파일
import multiprocessing

# 바인딩할 주소와 포트
bind = "0.0.0.0:8000"

# 워커 프로세스 수 (CPU 코어 수 * 2 + 1 권장)
workers = multiprocessing.cpu_count() * 2 + 1

# 워커 타입 (sync, gevent, eventlet 등)
worker_class = "sync"

# 타임아웃 설정 (초)
timeout = 120

# 최대 요청 수 (워커 재시작 전)
max_requests = 1000
max_requests_jitter = 100

# 로그 설정
accesslog = "-"
errorlog = "-"
loglevel = "info"

# 프로세스 이름
proc_name = "pdf_malware_detect"

# 데몬 모드 (백그라운드 실행)
daemon = False

# 사용자/그룹 (리눅스에서 권한 설정)
# user = "www-data"
# group = "www-data" 