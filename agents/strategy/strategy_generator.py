import openai
from typing import List, Dict, Optional, Any
import uuid
from datetime import datetime
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import json
import os
import logging
from .models import (
    StrategyType, TimeHorizon, MarketCondition,
    EntryCondition, ExitCondition, RiskParameters,
    TechnicalIndicator, StrategyResponse
)

# 로거 설정
logger = logging.getLogger(__name__)

class StrategyGenerator:
    def __init__(self):
        self.openai_client = openai.OpenAI()
        self.pipeline_dir = "data/pipeline"
        
    def _load_news_analysis(self) -> Dict:
        """뉴스 분석 결과 로드"""
        try:
            with open(f"{self.pipeline_dir}/news_output.json", 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return None

    def _analyze_market_conditions(self, news_data: Dict) -> MarketCondition:
        """뉴스 데이터를 기반으로 시장 상황 분석"""
        if not news_data:
            return None

        # 섹터별 감성 점수는 이미 계산되어 있음
        sector_performance = news_data.get('sector_sentiment', {})
        
        # 전체 시장 트렌드는 이미 계산된 감성 점수 활용
        avg_sentiment = np.mean([stock['감성점수'] for stock in news_data['stocks']])
        market_trend = "bullish" if avg_sentiment > 0.6 else "bearish" if avg_sentiment < 0.4 else "neutral"

        # 변동성 수준 판단 (감성 점수 기준)
        sentiment_std = np.std([stock['감성점수'] for stock in news_data['stocks']])
        volatility_level = "high" if sentiment_std > 0.3 else "low" if sentiment_std < 0.1 else "medium"

        return MarketCondition(
            market_trend=market_trend,
            volatility_level=volatility_level,
            trading_volume=np.mean([stock['뉴스갯수'] for stock in news_data['stocks']]),
            sector_performance=sector_performance,
            major_events=news_data.get('market_conditions', {}).get('major_events', []),
            timestamp=datetime.now()
        )

    def _select_strategy_type(self, market_conditions: MarketCondition) -> StrategyType:
        """시장 상황에 따른 전략 유형 선택"""
        if market_conditions.market_trend == "bullish":
            return StrategyType.MOMENTUM if market_conditions.volatility_level == "low" else StrategyType.TREND_FOLLOWING
        elif market_conditions.market_trend == "bearish":
            return StrategyType.MEAN_REVERSION if market_conditions.volatility_level == "high" else StrategyType.STATISTICAL_ARBITRAGE
        else:
            return StrategyType.BREAKOUT if market_conditions.volatility_level == "high" else StrategyType.MOMENTUM

    def _generate_technical_indicators(self, strategy_type: StrategyType, market_conditions: MarketCondition) -> List[TechnicalIndicator]:
        """전략 유형과 시장 상황에 따른 기술적 지표 생성"""
        indicators = []
        
        if strategy_type == StrategyType.MOMENTUM:
            indicators.extend([
                TechnicalIndicator(
                    name="RSI",
                    value=0.0,
                    signal="neutral",
                    parameters={"period": 14}
                ),
                TechnicalIndicator(
                    name="MACD",
                    value=0.0,
                    signal="neutral",
                    parameters={
                        "fast_period": 12,
                        "slow_period": 26,
                        "signal_period": 9
                    }
                )
            ])
        elif strategy_type == StrategyType.TREND_FOLLOWING:
            indicators.extend([
                TechnicalIndicator(
                    name="EMA",
                    value=0.0,
                    signal="neutral",
                    parameters={"period": 20}
                ),
                TechnicalIndicator(
                    name="ADX",
                    value=0.0,
                    signal="neutral",
                    parameters={"period": 14}
                )
            ])
        elif strategy_type == StrategyType.MEAN_REVERSION:
            indicators.extend([
                TechnicalIndicator(
                    name="Bollinger_Bands",
                    value=0.0,
                    signal="neutral",
                    parameters={
                        "period": 20,
                        "std_dev": 2
                    }
                ),
                TechnicalIndicator(
                    name="Stochastic",
                    value=0.0,
                    signal="neutral",
                    parameters={
                        "k_period": 14,
                        "d_period": 3
                    }
                )
            ])
            
        return indicators

    def _generate_entry_conditions(
        self,
        strategy_type: StrategyType,
        risk_tolerance: float,
        market_conditions: MarketCondition
    ) -> List[EntryCondition]:
        """시장 상황을 고려한 진입 조건 생성"""
        conditions = []
        
        if strategy_type == StrategyType.MOMENTUM:
            rsi_threshold = 30 if market_conditions.market_trend == "bullish" else 40
            conditions.extend([
                EntryCondition(
                    indicator="RSI",
                    condition="less_than",
                    threshold=float(rsi_threshold),
                    additional_params={"lookback_period": 14}
                ),
                EntryCondition(
                    indicator="MACD",
                    condition="crosses_above",
                    threshold=0.0,
                    additional_params={
                        "fast_period": 12,
                        "slow_period": 26
                    }
                ),
                EntryCondition(
                    indicator="News_Sentiment",
                    condition="greater_than",
                    threshold=0.6,
                    additional_params={"min_news_count": 5}
                )
            ])
        elif strategy_type == StrategyType.TREND_FOLLOWING:
            conditions.extend([
                EntryCondition(
                    indicator="EMA",
                    condition="price_above",
                    threshold=0.0,
                    additional_params={"period": 20}
                ),
                EntryCondition(
                    indicator="ADX",
                    condition="greater_than",
                    threshold=25.0,
                    additional_params={"period": 14}
                ),
                EntryCondition(
                    indicator="Sector_Sentiment",
                    condition="greater_than",
                    threshold=0.5,
                    additional_params={"lookback_days": 3}
                )
            ])
            
        return conditions

    def _generate_exit_conditions(
        self,
        strategy_type: StrategyType,
        risk_tolerance: float,
        market_conditions: MarketCondition
    ) -> List[ExitCondition]:
        """시장 상황을 고려한 청산 조건 생성"""
        conditions = []
        
        if strategy_type == StrategyType.MOMENTUM:
            rsi_threshold = 70 if market_conditions.market_trend == "bullish" else 60
            conditions.extend([
                ExitCondition(
                    indicator="RSI",
                    condition="greater_than",
                    threshold=float(rsi_threshold),
                    additional_params={"lookback_period": 14}
                ),
                ExitCondition(
                    indicator="Stop_Loss",
                    condition="less_than",
                    threshold=-0.02 * (1 + risk_tolerance)
                ),
                ExitCondition(
                    indicator="News_Sentiment",
                    condition="less_than",
                    threshold=0.4,
                    additional_params={"consecutive_days": 2}
                )
            ])
            
        return conditions

    def _generate_risk_parameters(
        self,
        risk_tolerance: float,
        strategy_type: StrategyType,
        market_conditions: MarketCondition
    ) -> RiskParameters:
        """시장 상황을 고려한 리스크 파라미터 생성"""
        # 변동성에 따른 포지션 크기 조정
        volatility_factor = {
            "low": 1.2,
            "medium": 1.0,
            "high": 0.8
        }[market_conditions.volatility_level]
        
        base_position_size = 0.1 * volatility_factor
        adjusted_position_size = base_position_size * (1 + risk_tolerance)
        
        # 시장 트렌드에 따른 손절/익절 조정
        trend_factor = {
            "bullish": 1.2,
            "neutral": 1.0,
            "bearish": 0.8
        }[market_conditions.market_trend]
        
        return RiskParameters(
            max_position_size=min(adjusted_position_size, 1.0),
            stop_loss=0.02 * (1 + risk_tolerance) * trend_factor,
            take_profit=0.04 * (1 + risk_tolerance) * trend_factor,
            max_drawdown=0.1 * (1 + risk_tolerance),
            risk_reward_ratio=2.0,
            max_correlation=0.7
        )

    def _select_target_assets(self, news_data: Dict, strategy_type: StrategyType, market_conditions: MarketCondition) -> List[str]:
        """뉴스 분석 결과를 기반으로 투자 대상 선정"""
        if not news_data:
            return []

        # 종목 선정 기준:
        # 1. 감성 점수가 높은 종목
        # 2. 뉴스 수가 많은 종목
        # 3. 시장 영향도가 높은 종목
        stocks_analysis = []
        for stock in news_data['stocks']:
            score = (
                stock['감성점수'] * 0.4 +
                min(stock['뉴스갯수'] / 20, 1) * 0.3 +
                abs(stock['시장영향도']) * 0.3
            )
            stocks_analysis.append((stock['종목명'], score))

        # 점수 기준으로 정렬하고 상위 5개 종목 선택
        stocks_analysis.sort(key=lambda x: x[1], reverse=True)
        return [stock[0] for stock in stocks_analysis[:5]]

    def _generate_stock_potential_analysis(self, stock_name: str, news_data: Dict, market_conditions: MarketCondition) -> str:
        """GPT를 활용하여 종목의 투자 유망성 분석"""
        # 해당 종목의 데이터 찾기
        stock_data = next((stock for stock in news_data['stocks'] if stock['종목명'] == stock_name), None)
        if not stock_data:
            return ""

        prompt = f"""
        다음 데이터를 기반으로 {stock_name}의 투자 유망성을 분석해주세요:
        
        1. 뉴스 감성 점수: {stock_data['감성점수']:.2f}
        2. 매수 확률: {stock_data['매수확률']}%
        3. 관련 뉴스 수: {stock_data['뉴스갯수']}건
        4. 시장 영향도: {stock_data['시장영향도']:.2f}
        5. 영향 받는 섹터: {', '.join(stock_data['영향섹터'])}
        6. 전반적인 시장 상황: {market_conditions.market_trend}
        7. 시장 변동성: {market_conditions.volatility_level}
        
        다음 형식으로 분석해주세요:
        1. 투자 포인트 (2-3줄)
        2. 위험 요소 (1-2줄)
        3. 향후 전망 (1-2줄)
        """

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"GPT 분석 실패: {str(e)}"

    def _calculate_sector_allocation(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """
        시장 데이터를 기반으로 섹터별 투자 비중을 계산합니다.
        
        Args:
            market_data: 시장 데이터 딕셔너리
            
        Returns:
            섹터별 투자 비중을 담은 딕셔너리 (섹터명: 비중)
        """
        sector_allocation = {}
        
        # 섹터 성과 데이터 추출
        sector_performance = market_data.get('sector_performance', {})
        if not sector_performance:
            logger.warning("섹터 성과 데이터가 없습니다. 기본 비중을 반환합니다.")
            return {'IT': 0.3, 'Healthcare': 0.2, 'Consumer': 0.2, 
                   'Financial': 0.15, 'Industrial': 0.15}
        
        # 섹터별 점수 계산
        sector_scores = {}
        for sector, data in sector_performance.items():
            momentum = data.get('momentum', 0)
            growth = data.get('growth', 0)
            value = data.get('value', 0)
            
            # 각 지표에 가중치를 적용하여 최종 점수 계산
            score = (momentum * 0.4) + (growth * 0.4) + (value * 0.2)
            sector_scores[sector] = max(score, 0)  # 음수 점수 방지
            
        # 점수 총합 계산
        total_score = sum(sector_scores.values())
        
        # 점수를 비중으로 변환
        if total_score > 0:
            for sector, score in sector_scores.items():
                sector_allocation[sector] = round(score / total_score, 2)
        else:
            logger.warning("모든 섹터의 점수가 0 이하입니다. 균등 비중을 적용합니다.")
            equal_weight = round(1.0 / len(sector_scores), 2)
            sector_allocation = {sector: equal_weight for sector in sector_scores}
        
        return sector_allocation

    def generate_strategy(
        self,
        user_input: str,
        market_conditions: Optional[MarketCondition] = None,
        risk_tolerance: Optional[float] = 0.5,
        time_horizon: Optional[TimeHorizon] = TimeHorizon.MEDIUM_TERM
    ) -> StrategyResponse:
        """전략 생성 메인 로직"""
        # 뉴스 분석 결과 로드
        news_data = self._load_news_analysis()
        
        # 시장 상황 분석
        if not market_conditions:
            market_conditions = self._analyze_market_conditions(news_data)
            
        # 섹터 비중 계산 (뉴스 데이터에서 시장 상황 데이터 추출)
        market_data = {
            'sector_performance': {
                sector: {
                    'momentum': news_data.get('sector_momentum', {}).get(sector, 0),
                    'growth': news_data.get('sector_growth', {}).get(sector, 0),
                    'value': news_data.get('sector_value', {}).get(sector, 0)
                }
                for sector in news_data.get('sector_sentiment', {}).keys()
            }
        }
        sector_allocation = self._calculate_sector_allocation(market_data)
        print("📊 자동 계산된 산업 비중:", sector_allocation)
        
        # 전략 유형 선택
        strategy_type = self._select_strategy_type(market_conditions)
        
        # 기술적 지표 생성
        technical_indicators = self._generate_technical_indicators(strategy_type, market_conditions)
        
        # 진입/청산 조건 생성
        entry_conditions = self._generate_entry_conditions(strategy_type, risk_tolerance, market_conditions)
        exit_conditions = self._generate_exit_conditions(strategy_type, risk_tolerance, market_conditions)
        
        # 리스크 파라미터 생성
        risk_parameters = self._generate_risk_parameters(risk_tolerance, strategy_type, market_conditions)
        
        # 투자 대상 선정
        target_assets = self._select_target_assets(news_data, strategy_type, market_conditions)
        
        # 종목별 유망성 분석
        stock_analyses = []
        for stock in target_assets:
            analysis = self._generate_stock_potential_analysis(stock, news_data, market_conditions)
            stock_analyses.append(f"\n[{stock} 투자 유망성 분석]\n{analysis}")
        
        # 전략 설명 생성
        explanation = f"""
        {strategy_type.value.upper()} 전략이 생성되었습니다.
        시장 상황: {market_conditions.market_trend} (변동성: {market_conditions.volatility_level})
        
        📊 산업별 투자 비중:
        {chr(10).join([f'- {sector}: {weight:.2%}' for sector, weight in sector_allocation.items()])}
        
        🎯 주요 투자 대상:
        {chr(10).join([f'- {stock}' for stock in target_assets])}
        
        📈 종목별 투자 유망성:
        {chr(10).join(stock_analyses)}
        
        ⚡ 진입 조건:
        - {''.join([f'{c.indicator}: {c.condition} {c.threshold}' for c in entry_conditions])}
        
        🔚 청산 조건:
        - {''.join([f'{c.indicator}: {c.condition} {c.threshold}' for c in exit_conditions])}
        
        ⚠️ 리스크 관리:
        - 최대 포지션 크기: {risk_parameters.max_position_size:.2%}
        - 손절: {risk_parameters.stop_loss:.2%}
        - 익절: {risk_parameters.take_profit:.2%}
        """
        
        # 전략 응답 생성
        strategy = StrategyResponse(
            strategy_id=f"STRAT_{uuid.uuid4().hex[:8]}",
            strategy_type=strategy_type,
            entry_conditions=entry_conditions,
            exit_conditions=exit_conditions,
            position_size=risk_parameters.max_position_size,
            risk_parameters=risk_parameters,
            technical_indicators=technical_indicators,
            target_assets=target_assets,
            time_horizon=time_horizon,
            explanation=explanation,
            sector_allocation=sector_allocation
        )
        
        return strategy 