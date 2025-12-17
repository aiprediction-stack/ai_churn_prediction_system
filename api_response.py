# utils/api_response.py
from flask import jsonify

class ApiResponse:
    @staticmethod
    def success(data=None, message="Success"):
        response = {
            "code": 200,
            "status": "success",
            "message": message,
            "data": data
        }
        return jsonify(response), 200

    @staticmethod
    def error(message="Error", code=400):
        response = {
            "code": code,
            "status": "error",
            "message": message,
            "data": None
        }
        return jsonify(response), code