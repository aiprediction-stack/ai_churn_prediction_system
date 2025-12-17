# config_v2.py
import os
from dotenv import load_dotenv

# 啟動時自動載入 .env
load_dotenv()

# 鎖定專案根目錄 (絕對路徑)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key'
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
    
    # 資料庫連線
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(BASE_DIR, 'app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # 模型路徑配置
    # 指向 projects/customer_churn_bank_code
    MODEL_ROOT = os.path.join(BASE_DIR, 'projects', 'customer_churn_bank_code')
    MODEL_BANK_PATH = os.path.join(MODEL_ROOT, 'customer_churn_bank_model.joblib')
    MODEL_META_DIR = MODEL_ROOT

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}