#!/usr/bin/env python3
"""
프로덕션 환경에서 Gunicorn으로 Flask 앱을 실행하는 스크립트
"""

import os
import sys
from app import app

if __name__ == "__main__":
    # 환경 변수 설정
    os.environ['FLASK_ENV'] = 'production'
    
    # Gunicorn으로 실행
    from gunicorn.app.wsgiapp import WSGIApplication
    
    class StandaloneApplication(WSGIApplication):
        def __init__(self, app, options=None):
            self.options = options or {}
            self.application = app
            super().__init__()
        
        def load_config(self):
            for key, value in self.options.items():
                self.cfg.set(key, value)
    
    # Gunicorn 옵션 설정
    options = {
        'bind': '0.0.0.0:8000',
        'workers': 4,
        'worker_class': 'sync',
        'timeout': 120,
        'max_requests': 1000,
        'max_requests_jitter': 100,
        'accesslog': '-',
        'errorlog': '-',
        'loglevel': 'info',
        'proc_name': 'pdf_malware_detect'
    }
    
    StandaloneApplication(app, options).run() 