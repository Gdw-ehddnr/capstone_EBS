from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime

class TimeHorizon(str, Enum):
    SHORT_TERM = "short_term"  # 1일 - 1주일
    MEDIUM_TERM = "medium_term"  # 1주일 - 1개월
    LONG_TERM = "long_term"  # 1개월 이상

class StrategyType(str, Enum):
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    TREND_FOLLOWING = "trend_following"
    BREAKOUT = "breakout"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    MACHINE_LEARNING = "machine_learning"
    VALUE = "value"
    GROWTH = "growth"

class MarketCondition(BaseModel):
    market_trend: str
    volatility_level: str
    trading_volume: float
    sector_performance: Dict[str, float]
    major_events: List[str]
    timestamp: datetime

class TechnicalIndicator(BaseModel):
    name: str
    value: float
    signal: str
    parameters: Dict[str, Any]

class EntryCondition(BaseModel):
    indicator: str
    condition: str
    threshold: float
    additional_params: Optional[Dict[str, Any]] = None

class ExitCondition(BaseModel):
    indicator: str
    condition: str
    threshold: float
    additional_params: Optional[Dict[str, Any]] = None

class RiskParameters(BaseModel):
    max_position_size: float = Field(..., gt=0, le=1)
    stop_loss: float = Field(..., gt=0, le=1)
    take_profit: Optional[float] = Field(None, gt=0)
    max_drawdown: float = Field(..., gt=0, le=1)
    risk_reward_ratio: float = Field(..., gt=0)
    max_correlation: Optional[float] = Field(None, ge=-1, le=1)

class StrategyRequest(BaseModel):
    user_input: str
    market_conditions: Optional[MarketCondition] = None
    risk_tolerance: Optional[float] = Field(None, ge=0, le=1)
    time_horizon: Optional[TimeHorizon] = None
    target_assets: Optional[List[str]] = None
    initial_capital: Optional[float] = None

class StrategyResponse(BaseModel):
    strategy_id: str
    strategy_type: StrategyType
    entry_conditions: List[EntryCondition]
    exit_conditions: List[ExitCondition]
    position_size: float
    risk_parameters: RiskParameters
    technical_indicators: List[TechnicalIndicator]
    target_assets: List[str]
    time_horizon: TimeHorizon
    explanation: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now) 
    sector_allocation: Optional[Dict[str, float]] = Field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "strategy_id": self.strategy_id,
            "strategy_type": self.strategy_type.value,
            "entry_conditions": [ec.__dict__ for ec in self.entry_conditions],
            "exit_conditions": [ec.__dict__ for ec in self.exit_conditions],
            "position_size": self.position_size,
            "risk_parameters": self.risk_parameters.__dict__,
            "technical_indicators": [ti.__dict__ for ti in self.technical_indicators],
            "target_assets": self.target_assets,
            "time_horizon": self.time_horizon.value,
            "explanation": self.explanation,
            "sector_allocation": self.sector_allocation or {}
        }

class FundamentalIndicators(BaseModel):
    pbr: float
    per: float
    dividend_yield: float
    debt_ratio: float
    revenue_growth: float
    operating_profit_growth: float
    roe: float
    last_updated: datetime

class FundamentalAnalysis(BaseModel):
    value_score: float
    growth_score: float
    analysis_summary: str
    recommendation: str
    confidence: float 