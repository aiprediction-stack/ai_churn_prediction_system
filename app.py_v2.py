# app_v2.py
import os
import time
import logging
from flask import Flask, render_template, request, g
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from config_v2 import config
from utils.api_response import ApiResponse

db = SQLAlchemy()
# 備註：這裡的 logging.basicConfig(level=logging.INFO) 會被 configure_logging 覆蓋，但保留不影響。
#logging.basicConfig(level=logging.INFO)
gateway_logger = logging.getLogger('API_Gateway')


def configure_logging(app):
    """
    確保所有 INFO 級別的日誌都能輸出到終端機。
    """
    # 這裡可以直接使用 logging.basicConfig 來配置根日誌處理器
    # 這樣在 app.run() 啟動前，所有 INFO 級別的日誌都會被輸出。
    if app.debug:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

def create_app(config_name='default'):
    app = Flask(__name__)

    # 1. 配置載入 (必須在日誌配置前)
    app.config.from_object(config[config_name])
    app.debug = app.config.get('DEBUG', False)

    # 2. 🚨 關鍵：日誌配置 (確保 INFO 級別能顯示)
    # 這裡的 app.debug 需要正確反映 config[config_name] 的設定
    app.debug = app.config.get('DEBUG', False)
    configure_logging(app)
    
    CORS(app)
    db.init_app(app)

    @app.before_request
    def gateway_inspection():
        g.start_time = time.time()
        if request.path.startswith('/static') or request.path == '/': return None
        ip = request.remote_addr
        gateway_logger.info(f"🚧 [Inbound] {request.method} {request.path} from {ip}")
        # 隱形模式：暫不阻擋 API Key

    @app.after_request
    def gateway_logging(response):
        if hasattr(g, 'start_time'):
            elapsed = time.time() - g.start_time
            msg = f"✅ [Outbound] Status: {response.status_code} | Time: {elapsed:.4f}s"
            if response.status_code >= 400: gateway_logger.error(msg)
            else: gateway_logger.info(msg)
        return response
    
# 3. 🎯 核心改變 2：延後導入路由 (確保日誌配置已完成)
    from routes.customer_churn_bank_routes_v2 import customer_churn_bank_blueprint
    app.register_blueprint(customer_churn_bank_blueprint, url_prefix='/api/customer_churn_bank')

    @app.route('/')
    def index(): return render_template('index.html')

    @app.route('/customer_churn_bank_model')
    def customer_churn_bank_page(): # <--- 改回這個長名字！
        return render_template('customer_churn_bank.html')
        
    return app

if __name__ == '__main__':
    env = os.getenv('FLASK_CONFIG', 'default')
    app = create_app(env)
    port = int(os.environ.get('PORT', 5000))
    print(f"🔥 V2 Gateway Launched | Mode: {env}")
    app.run(host='0.0.0.0', port=port)