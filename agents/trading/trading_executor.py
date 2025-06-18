import uuid
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
from decimal import Decimal
import yfinance as yf
import os
from .models import (
    OrderType, OrderSide, OrderStatus, TimeInForce,
    PositionSide, PositionStatus, Order, Position,
    RiskMetrics, ExecutionRequest, ExecutionResponse,
    PositionUpdateRequest, PositionUpdateResponse,
    OrderCancelRequest, OrderCancelResponse,
    PortfolioStatus, PositionInfo, RiskLimits,
    OrderValidationError, RiskLimitExceededError, OrderExecutionError, PortfolioError
)

class TradingExecutor:
    def __init__(self):
        self.positions: Dict[str, Position] = {}  # ticker -> Position
        self.orders: Dict[str, Order] = {}        # order_id -> Order
        self.cash_balance = Decimal("1000000")  # 초기 자본금
        self.risk_limits = RiskLimits(
            max_position_size=Decimal("100000"),  # 10만 달러
            max_total_exposure=Decimal("500000"),  # 50만 달러
            max_drawdown=Decimal("0.1"),
            var_limit=Decimal("0.05")
        )
        self.max_position_size = 1000000  # $1M
        self.max_exposure = 5000000       # $5M
        
    def _generate_order_id(self) -> str:
        """주문 ID 생성"""
        return str(uuid.uuid4())
        
    def _generate_position_id(self) -> str:
        """포지션 ID 생성"""
        return str(uuid.uuid4())
        
    def _get_current_price(self, ticker: str) -> Decimal:
        """현재 가격 조회"""
        try:
            # 테스트 환경에서는 고정된 가격 반환
            if os.getenv("TESTING") == "true":
                # 테스트를 위해 현재가를 98.0으로 설정
                return Decimal("98.0")
                
            stock = yf.Ticker(ticker)
            current_price = stock.info['regularMarketPrice']
            return Decimal(str(current_price))
        except Exception as e:
            raise Exception(f"Failed to fetch price for {ticker}: {str(e)}")
            
    def _calculate_commission(self, price: Decimal, quantity: Decimal) -> Decimal:
        """수수료 계산"""
        commission_rate = Decimal("0.00025")  # 0.025%
        return price * quantity * commission_rate
        
    def _calculate_risk_metrics(self, position: Position) -> RiskMetrics:
        """리스크 메트릭스 계산"""
        position_value = abs(position.quantity * position.current_price)
        return RiskMetrics(
            var=float(position_value * self.risk_limits.var_limit),
            drawdown=float(position_value * self.risk_limits.max_drawdown),
            exposure=float(position_value),
            leverage=float(position_value / self.cash_balance)
        )
        
    def _validate_risk_limits(self, order: Order, current_price: Decimal) -> Tuple[bool, str]:
        """리스크 한도 검증"""
        try:
            # 주문 크기 계산 (금액 기준)
            order_value = order.quantity * current_price
            
            # 최대 포지션 크기 검증
            if order_value > self.risk_limits.max_position_size:
                return False, f"Position size ({float(order_value)}) exceeds maximum limit ({float(self.risk_limits.max_position_size)})"
            
            # 총 노출도 계산
            total_exposure = Decimal('0')
            for pos in self.positions.values():
                if pos.status == "open":
                    total_exposure += abs(pos.quantity * pos.current_price)
            
            # 새 포지션 추가
            total_exposure += order_value
            
            # 총 노출도 검증
            if total_exposure > self.risk_limits.max_total_exposure:
                return False, f"Total exposure ({float(total_exposure)}) exceeds maximum limit ({float(self.risk_limits.max_total_exposure)})"
            
            return True, ""
            
        except Exception as e:
            return False, f"Risk validation error: {str(e)}"
        
    def _update_position_pnl(self, position: Position):
        """포지션 손익 업데이트"""
        if position.status == PositionStatus.OPEN:
        # 미실현 손익 계산
        if position.side == PositionSide.LONG:
                position.unrealized_pnl = (position.current_price - position.entry_price) * position.quantity
            else:
                position.unrealized_pnl = (position.entry_price - position.current_price) * position.quantity
        else:
            # 실현 손익 계산
            if position.side == PositionSide.LONG:
                position.realized_pnl = (position.current_price - position.entry_price) * position.quantity
            else:
                position.realized_pnl = (position.entry_price - position.current_price) * position.quantity
            position.unrealized_pnl = Decimal("0")

    def _calculate_execution_price(self, order: Order, current_price: Decimal) -> Optional[Decimal]:
        """주문 실행 가격 계산"""
        if order.order_type == OrderType.MARKET:
            return current_price
            
        elif order.order_type == OrderType.LIMIT:
            # 매수 주문: 현재가가 지정가보다 높거나 같을 때 체결
            if order.side == OrderSide.BUY:
                if current_price >= order.price:
                    return current_price  # 현재가로 체결
            # 매도 주문: 현재가가 지정가보다 낮거나 같을 때 체결
            else:
                if current_price <= order.price:
                    return current_price  # 현재가로 체결
                
        elif order.order_type == OrderType.STOP:
            # 매수 주문: 현재가가 스탑가보다 높거나 같을 때 체결
            if order.side == OrderSide.BUY:
                if current_price >= order.stop_price:
                    return current_price  # 시장가로 체결
            # 매도 주문: 현재가가 스탑가보다 낮거나 같을 때 체결
            else:
                if current_price <= order.stop_price:
                    return current_price  # 시장가로 체결
                
        return None

    def _validate_order(self, order: Order) -> bool:
        """주문 유효성 검증"""
        try:
            if order.quantity <= 0:
            return False
        
            if order.order_type == OrderType.LIMIT:
                if order.price is None or order.price <= 0:
            return False
            
            if order.order_type == OrderType.STOP:
                if order.stop_price is None or order.stop_price <= 0:
            return False
            
        return True
            
        except Exception:
            return False

    def _check_risk_limits(self, order: Order) -> bool:
        """리스크 한도 검사"""
        current_position = self.positions.get(order.symbol, None)
        position_value = abs(order.quantity * order.market_price)
        
        if current_position:
            new_position_size = abs(current_position.quantity + order.quantity) * order.market_price
        else:
            new_position_size = position_value
            
        total_exposure = sum(abs(pos.quantity * pos.avg_price) for pos in self.positions.values())
        new_total_exposure = total_exposure + position_value
        
        return new_position_size <= self.max_position_size and new_total_exposure <= self.max_exposure

    def _update_position(self, order: Order, execution_price: float) -> Position:
        """포지션 업데이트"""
        position = self.positions.get(order.symbol)
        
        if position is None:
            position = Position(
                position_id=str(uuid.uuid4()),
                symbol=order.symbol,
                quantity=order.quantity if order.side == OrderSide.BUY else -order.quantity,
                avg_price=execution_price,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                timestamp=datetime.now().isoformat()
            )
        else:
            # Update position quantity and average price
            old_quantity = position.quantity
            old_avg_price = position.avg_price
            
            if order.side == OrderSide.BUY:
                new_quantity = old_quantity + order.quantity
            else:
                new_quantity = old_quantity - order.quantity
                
            if new_quantity != 0:
                position.avg_price = ((old_quantity * old_avg_price) + (order.quantity * execution_price)) / abs(new_quantity)
            position.quantity = new_quantity
            
        self.positions[order.symbol] = position
        return position

    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """주문 취소"""
        if order_id not in self.orders:
            return {
                "status": "ERROR",
                "message": f"Order {order_id} not found"
            }
            
        order = self.orders[order_id]
        if order.status != OrderStatus.PENDING:
            return {
                "status": "ERROR",
                "message": f"Order {order_id} cannot be cancelled"
            }
            
        order.status = OrderStatus.CANCELLED
        return {
            "status": "SUCCESS",
            "message": f"Order {order_id} cancelled successfully"
        }

    async def get_portfolio_status(self) -> PortfolioStatus:
        """포트폴리오 상태 조회"""
        try:
            positions = []
            total_value = self.cash_balance
        
            # 포지션 정보 업데이트
        for position in self.positions.values():
                if position.status == "open":
                    # 현재 가격 업데이트
                    current_price = self._get_current_price(position.ticker)
                    position.current_price = current_price
                    
                    # 손익 계산
                    self._update_position_pnl(position)
                    
                    # 포지션 가치 계산
                    position_value = position.quantity * position.current_price
            total_value += position_value
                    
                    positions.append(position)
            
            # 리스크 메트릭스 계산
            risk_metrics = {
                "var": 0.0,
                "drawdown": 0.0,
                "exposure": 0.0,
                "leverage": 0.0
            }
            if positions:
                metrics = self._calculate_risk_metrics(positions[0])
                risk_metrics = {
                    "var": float(metrics.var),
                    "drawdown": float(metrics.drawdown),
                    "exposure": float(metrics.exposure),
                    "leverage": float(metrics.leverage)
                }
            
            return PortfolioStatus(
                portfolio_id=str(uuid.uuid4()),
                cash_balance=float(self.cash_balance),
                total_value=float(total_value),
                positions=[{
                    "position_id": pos.position_id,
                    "ticker": pos.ticker,
                    "quantity": float(pos.quantity),
                    "entry_price": float(pos.entry_price),
                    "current_price": float(pos.current_price),
                    "unrealized_pnl": float(pos.unrealized_pnl),
                    "realized_pnl": float(pos.realized_pnl),
                    "side": pos.side.value,
                    "status": pos.status,
                    "timestamp": pos.timestamp.isoformat()
                } for pos in positions],
                risk_metrics=risk_metrics,
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            print(f"Error in get_portfolio_status: {str(e)}")  # 디버깅을 위한 로그 추가
            raise Exception(f"Failed to get portfolio status: {str(e)}")

    def _create_or_update_position(self, order: Order, execution_price: Decimal) -> Position:
        """포지션 생성 또는 업데이트"""
        # 기존 포지션 찾기
        position = self.positions.get(order.ticker)
        
        if position is None:
            # 새 포지션 생성
        position = Position(
                position_id=str(uuid.uuid4()),
            ticker=order.ticker,
            quantity=order.quantity,
                entry_price=execution_price,
                current_price=execution_price,
                side=order.side,
                status="open",
                timestamp=datetime.now()
            )
        else:
            # 기존 포지션 업데이트
            if position.side == order.side:
                # 같은 방향의 거래: 평균 진입가격 계산
                total_quantity = position.quantity + order.quantity
                position.entry_price = ((position.quantity * position.entry_price) + 
                                     (order.quantity * execution_price)) / total_quantity
                position.quantity = total_quantity
            else:
                # 반대 방향의 거래: 수량 감소
                if position.quantity > order.quantity:
                    position.quantity -= order.quantity
                else:
                    # 포지션 종료 또는 반대 방향으로 전환
                    remaining_quantity = order.quantity - position.quantity
                    if remaining_quantity == 0:
                        position.status = "closed"
                    else:
                        position.quantity = remaining_quantity
                        position.side = order.side
                        position.entry_price = execution_price
            
            position.current_price = execution_price
            position.timestamp = datetime.now()
            
        return position
            
    def _create_position_info(self, position: Position) -> dict:
        """포지션 정보 생성"""
        return {
            "position_id": position.position_id,
            "ticker": position.ticker,
            "quantity": float(position.quantity),
            "entry_price": float(position.entry_price),
            "current_price": float(position.current_price),
            "unrealized_pnl": float(position.unrealized_pnl),
            "realized_pnl": float(position.realized_pnl),
            "side": position.side.value,
            "status": position.status,
            "timestamp": position.timestamp.isoformat()
        }

    async def update_position(self, request: PositionUpdateRequest) -> PositionUpdateResponse:
        """포지션 업데이트"""
        try:
            position = None
            for pos in self.positions.values():
                if pos.position_id == request.position_id:
                    position = pos
                    break

            if not position:
                return PositionUpdateResponse(
                    request_id=str(uuid.uuid4()),
                    status="error",
                    message=f"Position not found: {request.position_id}",
                    position=None
                )

            # 포지션 정보 업데이트
            if "price" in request.updates:
                position.current_price = Decimal(str(request.updates["price"]))
            if "quantity" in request.updates:
                position.quantity = Decimal(str(request.updates["quantity"]))
            if "ticker" in request.updates:
                position.ticker = request.updates["ticker"]
                
            self._update_position_pnl(position)
                    
            return PositionUpdateResponse(
                request_id=str(uuid.uuid4()),
                status="success",
                message="Position updated successfully",
                position=position
            )
            
        except Exception as e:
            return PositionUpdateResponse(
                request_id=str(uuid.uuid4()),
                status="error",
                message=f"Failed to update position: {str(e)}",
                position=None
            )

    async def execute_order(self, order: Order) -> ExecutionResponse:
        """주문 실행"""
        try:
            # 주문 유효성 검증
            if not self._validate_order(order):
                return ExecutionResponse(
                    order_id=self._generate_order_id(),
                    status=OrderStatus.REJECTED,
                    message="Invalid order parameters",
                    ticker=order.ticker,
                    quantity=float(order.quantity),
                    price=float(order.price) if order.price else None,
                    position_info=None
                )

            # 현재 가격 조회
            current_price = self._get_current_price(order.ticker)
            
            # 실행 가격 계산
            execution_price = self._calculate_execution_price(order, current_price)
            if execution_price is None:
                return ExecutionResponse(
                    order_id=self._generate_order_id(),
                    status=OrderStatus.REJECTED,
                    message="Order conditions not met",
                    ticker=order.ticker,
                    quantity=float(order.quantity),
                    price=float(order.price) if order.price else None,
                    position_info=None
                )

            # 리스크 한도 검증
            risk_valid, risk_message = self._validate_risk_limits(order, execution_price)
            if not risk_valid:
                return ExecutionResponse(
                    order_id=self._generate_order_id(),
                    status=OrderStatus.REJECTED,
                    message=risk_message,
                    ticker=order.ticker,
                    quantity=float(order.quantity),
                    price=float(order.price) if order.price else None,
                    position_info=None
                )

            # 포지션 생성 또는 업데이트
            try:
                position = self._create_or_update_position(order, execution_price)
                # 포지션 저장 - ticker를 키로 사용
                self.positions[order.ticker] = position
                position_info = self._create_position_info(position)
                
                return ExecutionResponse(
                    order_id=self._generate_order_id(),
                    status=OrderStatus.FILLED,
                    message="Order executed successfully",
                    ticker=order.ticker,
                    quantity=float(order.quantity),
                    price=float(order.price) if order.price else None,
                    execution_price=float(execution_price),
                    position_info=position_info
                )
            except Exception as e:
                return ExecutionResponse(
                    order_id=self._generate_order_id(),
                    status=OrderStatus.REJECTED,
                    message=f"Position update failed: {str(e)}",
                    ticker=order.ticker,
                    quantity=float(order.quantity),
                    price=float(order.price) if order.price else None,
                    position_info=None
                )
                
        except Exception as e:
            return ExecutionResponse(
                order_id=self._generate_order_id(),
                status=OrderStatus.REJECTED,
                message=f"Order execution failed: {str(e)}",
                ticker=order.ticker if order else None,
                quantity=float(order.quantity) if order else None,
                price=float(order.price) if order and order.price else None,
                position_info=None
            ) 