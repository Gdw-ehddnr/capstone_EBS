from datetime import datetime
from fastapi.testclient import TestClient
from agents.trading.api import app
from agents.trading.models import ExecutionRequest

client = TestClient(app)

def generate_test_execution_request() -> dict:
    """테스트용 주문 실행 요청 생성"""
    return {
        "ticker": "AAPL",
        "quantity": 100,
        "price": 150.0,
        "order_type": "MARKET",
        "side": "BUY",
        "time_in_force": "GTC"
    }

def test_execute_order_endpoint():
    """주문 실행 엔드포인트 테스트"""
    request_data = generate_test_execution_request()
    response = client.post("/execute", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert "order_id" in data
    assert "ticker" in data
    assert "quantity" in data
    assert "price" in data
    assert "status" in data
    assert "position_info" in data

def test_execute_limit_order():
    """지정가 주문 실행 테스트"""
    request_data = generate_test_execution_request()
    request_data.update({
        "order_type": "LIMIT",
        "price": 98.0  # 현재가와 같은 가격
    })

    response = client.post("/execute", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "FILLED"
    assert data["price"] == 98.0

def test_execute_stop_order():
    """스탑 주문 실행 테스트"""
    request_data = generate_test_execution_request()
    request_data.update({
        "order_type": "STOP",
        "stop_price": 102.0  # 현재가보다 높은 가격
    })

    response = client.post("/execute", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "REJECTED"  # 현재가가 스탑가보다 낮으므로 체결되지 않음
    assert "Order conditions not met" in data["message"]

def test_update_position():
    """포지션 업데이트 테스트"""
    # 먼저 포지션 생성
    exec_request = generate_test_execution_request()
    exec_response = client.post("/execute", json=exec_request)
    position_info = exec_response.json()["position_info"]

    # 포지션 업데이트
    update_request = {
        "position_id": position_info["position_id"],
        "action": "modify",
        "updates": {
        "ticker": "AAPL",
        "quantity": 50,
            "price": 155.0
        }
    }
    response = client.post("/position/update", json=update_request)
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "success"

def test_cancel_order():
    """주문 취소 테스트"""
    # 먼저 지정가 주문 생성
    exec_request = generate_test_execution_request()
    exec_request.update({
        "order_type": "LIMIT",
        "price": 98.0
    })
    exec_response = client.post("/execute", json=exec_request)
    order_id = exec_response.json()["order_id"]

    # 주문 취소
    response = client.post(f"/order/cancel/{order_id}")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "success"
    assert data["message"] == f"Order {order_id} cancelled successfully"

def test_get_portfolio_status():
    """포트폴리오 상태 조회 테스트"""
    response = client.get("/portfolio")
    assert response.status_code == 200

    data = response.json()
    assert "portfolio_id" in data
    assert "total_value" in data
    assert "cash_balance" in data
    assert "positions" in data
    assert "risk_metrics" in data

def test_risk_limits():
    """리스크 한도 검증 테스트"""
    # 최대 포지션 크기를 초과하는 주문
    request_data = generate_test_execution_request()
    request_data["quantity"] = 2000  # max_position_size보다 큰 수량

    response = client.post("/execute", json=request_data)
    assert response.status_code == 400  # Bad Request

    data = response.json()
    assert "detail" in data
    assert "position size" in data["detail"].lower()

def test_position_pnl_calculation():
    """포지션 손익 계산 테스트"""
    # 먼저 포지션 생성
    exec_request = generate_test_execution_request()
    exec_response = client.post("/execute", json=exec_request)
    position_info = exec_response.json()["position_info"]
    print("\nExecute Response:", exec_response.json())
    print("Position Info:", position_info)

    # 포트폴리오 상태 조회
    response = client.get("/portfolio")
    assert response.status_code == 200

    data = response.json()
    positions = data["positions"]
    print("Portfolio Positions:", positions)
    
    position = next(p for p in positions if p["position_id"] == position_info["position_id"])
    assert "unrealized_pnl" in position 