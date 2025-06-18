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
from dotenv import load_dotenv
from .models import (
    StrategyType, TimeHorizon, MarketCondition,
    EntryCondition, ExitCondition, RiskParameters,
    TechnicalIndicator, StrategyResponse,
    FundamentalIndicators, FundamentalAnalysis
)
from agents.utils.call_openai_api import call_openai_api

# 로거 설정
logger = logging.getLogger(__name__)

class StrategyGenerator:
    def __init__(self):
        # .env 파일 로드
        load_dotenv()
        
        # OpenAI API 키 설정
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")
        
        self.openai_client = openai.OpenAI(api_key=api_key)
        self.pipeline_dir = "data/pipeline"
        
    def _log_openai_response(self, response: Any):
        """OpenAI 응답을 로깅합니다."""
        try:
            if hasattr(response, 'choices'):
                logger.info(f"OpenAI Response: {response.choices[0].message.content}")
            else:
                logger.info(f"OpenAI Response: {response}")
        except Exception as e:
            logger.error(f"Error logging OpenAI response: {str(e)}")
            
    def _handle_openai_error(self, error: Exception):
        """OpenAI 오류를 처리하고 로깅합니다."""
        error_message = str(error)
        logger.error(f"OpenAI API Error: {error_message}")
        if "authentication" in error_message.lower():
            logger.error("OpenAI API 키가 올바르게 설정되지 않았습니다. OPENAI_API_KEY 환경 변수를 확인해주세요.")
        raise

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
        """시장 상황에 따른 전략 유형 선택
        
        전략 선택 기준:
        1. Value 전략:
           - 시장이 하락세(bearish)이고 변동성이 낮을 때
           - PER, PBR이 낮고 배당수익률이 높은 기업 대상
           
        2. Growth 전략:
           - 시장이 상승세(bullish)이고 변동성이 낮을 때
           - 매출/영업이익 성장률이 높고 ROE가 좋은 기업 대상
           
        3. Momentum 전략:
           - 시장이 상승세이고 변동성이 중간 정도일 때
           - 최근 수익률이 좋고 거래량이 증가하는 기업 대상
        """
        if market_conditions.market_trend == "bullish":
            if market_conditions.volatility_level == "low":
                return StrategyType.GROWTH
            elif market_conditions.volatility_level == "medium":
                return StrategyType.MOMENTUM
            else:
                return StrategyType.TREND_FOLLOWING
        elif market_conditions.market_trend == "bearish":
            if market_conditions.volatility_level == "low":
                return StrategyType.VALUE
            elif market_conditions.volatility_level == "high":
                return StrategyType.MEAN_REVERSION
            else:
                return StrategyType.STATISTICAL_ARBITRAGE
        else:  # neutral market
            if market_conditions.volatility_level == "high":
                return StrategyType.BREAKOUT
            elif market_conditions.volatility_level == "low":
                return StrategyType.VALUE
            else:
                return StrategyType.MOMENTUM

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
            self._log_openai_response(response)
            return response.choices[0].message.content
        except Exception as e:
            self._handle_openai_error(e)
            return f"GPT 분석 실패: {str(e)}"

    def _analyze_fundamentals(self, stock_code: str) -> FundamentalIndicators:
        """기업의 재무제표를 분석하여 기본적 지표들을 계산"""
        try:
            # OpenAI API를 통해 최신 재무제표 데이터 요청
            prompt = f"""
            {stock_code} 기업의 최신 재무제표를 분석하여 다음 지표들을 계산해주세요:
            1. PBR (주가순자산비율)
            2. PER (주가수익비율)
            3. 배당수익률
            4. 부채비율
            5. 매출 성장률 (전년 대비)
            6. 영업이익 성장률 (전년 대비)
            7. ROE (자기자본이익률)
            
            각 지표의 수치와 함께 간단한 설명을 제공해주세요.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            # GPT의 응답을 파싱하여 FundamentalIndicators 객체 생성
            # 실제 구현시에는 더 정교한 파싱 로직이 필요합니다
            return FundamentalIndicators(
                pbr=float(response.choices[0].message.content.split('PBR:')[1].split('\n')[0]),
                per=float(response.choices[0].message.content.split('PER:')[1].split('\n')[0]),
                dividend_yield=float(response.choices[0].message.content.split('배당수익률:')[1].split('\n')[0]),
                debt_ratio=float(response.choices[0].message.content.split('부채비율:')[1].split('\n')[0]),
                revenue_growth=float(response.choices[0].message.content.split('매출성장률:')[1].split('\n')[0]),
                operating_profit_growth=float(response.choices[0].message.content.split('영업이익성장률:')[1].split('\n')[0]),
                roe=float(response.choices[0].message.content.split('ROE:')[1].split('\n')[0]),
                last_updated=datetime.now()
            )
        except Exception as e:
            logger.error(f"기본적 지표 분석 중 오류 발생: {str(e)}")
            raise

    def _analyze_value_strategy(self, indicators: FundamentalIndicators) -> FundamentalAnalysis:
        """Value 투자 전략 분석"""
        try:
            prompt = f"""
            다음 기업의 가치 투자 관점에서 분석해주세요:
            
            1. PBR: {indicators.pbr}
            2. PER: {indicators.per}
            3. 배당수익률: {indicators.dividend_yield}%
            4. 부채비율: {indicators.debt_ratio}%
            
            가치투자 관점에서 이 기업의 매력도를 0~1 사이의 점수로 평가하고,
            투자 추천 의견(매수/매도/보유)과 그 이유를 설명해주세요.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            analysis = response.choices[0].message.content
            # 응답 파싱 (실제 구현시에는 더 정교한 파싱 로직 필요)
            value_score = float(analysis.split('점수:')[1].split('\n')[0])
            recommendation = analysis.split('추천:')[1].split('\n')[0]
            
            return FundamentalAnalysis(
                value_score=value_score,
                growth_score=0.0,  # Value 분석에서는 사용하지 않음
                analysis_summary=analysis,
                recommendation=recommendation,
                confidence=0.8 if value_score > 0.7 else 0.5
            )
        except Exception as e:
            logger.error(f"Value 전략 분석 중 오류 발생: {str(e)}")
            raise

    def _analyze_growth_strategy(self, indicators: FundamentalIndicators) -> FundamentalAnalysis:
        """Growth 투자 전략 분석"""
        try:
            prompt = f"""
            다음 기업의 성장 투자 관점에서 분석해주세요:
            
            1. 매출 성장률: {indicators.revenue_growth}%
            2. 영업이익 성장률: {indicators.operating_profit_growth}%
            3. ROE: {indicators.roe}%
            
            성장투자 관점에서 이 기업의 매력도를 0~1 사이의 점수로 평가하고,
            투자 추천 의견(매수/매도/보유)과 그 이유를 설명해주세요.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            analysis = response.choices[0].message.content
            # 응답 파싱
            growth_score = float(analysis.split('점수:')[1].split('\n')[0])
            recommendation = analysis.split('추천:')[1].split('\n')[0]
            
            return FundamentalAnalysis(
                value_score=0.0,  # Growth 분석에서는 사용하지 않음
                growth_score=growth_score,
                analysis_summary=analysis,
                recommendation=recommendation,
                confidence=0.8 if growth_score > 0.7 else 0.5
            )
        except Exception as e:
            logger.error(f"Growth 전략 분석 중 오류 발생: {str(e)}")
            raise

    def _generate_fundamental_strategy(self, stock_code: str) -> Dict[str, Any]:
        """기본적 분석 기반의 종합 투자 전략 생성"""
        # 기본적 지표 분석
        indicators = self._analyze_fundamentals(stock_code)
        
        # Value 전략 분석
        value_analysis = self._analyze_value_strategy(indicators)
        
        # Growth 전략 분석
        growth_analysis = self._analyze_growth_strategy(indicators)
        
        # 종합 분석
        combined_score = (value_analysis.value_score + growth_analysis.growth_score) / 2
        confidence = (value_analysis.confidence + growth_analysis.confidence) / 2
        
        return {
            "indicators": indicators.dict(),
            "value_analysis": value_analysis.dict(),
            "growth_analysis": growth_analysis.dict(),
            "combined_score": combined_score,
            "confidence": confidence,
            "final_recommendation": "BUY" if combined_score > 0.7 else "HOLD" if combined_score > 0.4 else "SELL"
        }

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
        
        # 전략 유형 선택
        strategy_type = self._select_strategy_type(market_conditions)
        
        # 기술적 지표 또는 기본적 지표 생성
        if strategy_type in [StrategyType.VALUE, StrategyType.GROWTH]:
            # 기본적 분석 전략 실행
            target_assets = self._select_target_assets(news_data, strategy_type, market_conditions)
            fundamental_analyses = {}
            for asset in target_assets:
                fundamental_analyses[asset] = self._generate_fundamental_strategy(asset)
            
            # 리스크 파라미터 생성
            risk_parameters = self._generate_risk_parameters(risk_tolerance, strategy_type, market_conditions)
            
            # 분석 결과를 설명 문자열로 변환
            explanation = f"""
            기본적 분석 전략 ({strategy_type.value.upper()}) 결과:
            
            분석 대상 종목:
            {chr(10).join([f'- {asset}: {analysis["final_recommendation"]} (신뢰도: {analysis["confidence"]:.2f})'
                          for asset, analysis in fundamental_analyses.items()])}
            
            주요 분석 내용:
            {chr(10).join([f'[{asset}]\n{analysis["value_analysis"]["analysis_summary"]}\n{analysis["growth_analysis"]["analysis_summary"]}'
                          for asset, analysis in fundamental_analyses.items()])}
            """
        else:
            # 기존 기술적 분석 로직 실행
            technical_indicators = self._generate_technical_indicators(strategy_type, market_conditions)
            entry_conditions = self._generate_entry_conditions(strategy_type, risk_tolerance, market_conditions)
            exit_conditions = self._generate_exit_conditions(strategy_type, risk_tolerance, market_conditions)
            target_assets = self._select_target_assets(news_data, strategy_type, market_conditions)
            
            explanation = f"""
            기술적 분석 전략 ({strategy_type.value.upper()}) 결과:
            ...기존 설명 로직...
            """
        
        # 전략 응답 생성
        strategy = StrategyResponse(
            strategy_id=f"STRAT_{uuid.uuid4().hex[:8]}",
            strategy_type=strategy_type,
            entry_conditions=entry_conditions if strategy_type not in [StrategyType.VALUE, StrategyType.GROWTH] else [],
            exit_conditions=exit_conditions if strategy_type not in [StrategyType.VALUE, StrategyType.GROWTH] else [],
            position_size=risk_parameters.max_position_size if 'risk_parameters' in locals() else 0.1,
            risk_parameters=risk_parameters if 'risk_parameters' in locals() else RiskParameters(
                max_position_size=0.1,
                stop_loss=0.05,
                take_profit=0.1,
                max_drawdown=0.2,
                risk_reward_ratio=2.0,
                max_correlation=0.7
            ),
            technical_indicators=technical_indicators if 'technical_indicators' in locals() else [],
            target_assets=target_assets,
            time_horizon=time_horizon,
            explanation=explanation
        )
        
        return strategy

    def propose(self, context):
        """
        자신의 전략 결과를 의견으로 제시합니다.
        """
        market_conditions = context.get('market_conditions', None)
        strategy = self.generate_strategy(user_input="", market_conditions=market_conditions)
        trend = market_conditions.market_trend if market_conditions else 'neutral'
        volatility = getattr(market_conditions, 'volatility_level', 'medium') if market_conditions else 'medium'
        strategy_type = getattr(strategy, 'strategy_type', None)
        decision = 'HOLD'
        confidence = 0.5
        reasons = []
        if strategy_type and hasattr(strategy_type, 'value'):
            stype = strategy_type.value.lower()
            if stype == 'momentum' and trend == 'bullish' and volatility == 'low':
                decision = 'BUY'
                confidence = 0.8
                reasons.append('모멘텀+상승장+저변동성')
            elif stype == 'trend_following' and trend == 'bullish':
                decision = 'BUY'
                confidence = 0.7
                reasons.append('추세추종+상승장')
            elif stype == 'mean_reversion' and trend == 'bearish' and volatility == 'high':
                decision = 'SELL'
                confidence = 0.8
                reasons.append('역추세+하락장+고변동성')
            elif stype == 'statistical_arbitrage' and trend == 'bearish':
                decision = 'SELL'
                confidence = 0.7
                reasons.append('통계차익+하락장')
            elif stype == 'breakout' and volatility == 'high':
                decision = 'BUY'
                confidence = 0.6
                reasons.append('돌파+고변동성')
            else:
                decision = 'HOLD'
                confidence = 0.5
                reasons.append(f'전략: {stype}, 트렌드: {trend}, 변동성: {volatility}')
        reason = ', '.join(reasons) if reasons else '전략 분석 결과'
        return {
            'agent': 'strategy_generator',
            'decision': decision,
            'confidence': confidence,
            'reason': reason
        }

    def debate(self, context, others_opinions, my_opinion_1st_round=None):
        market_conditions = context.get('market_conditions', None)
        strategy = self.generate_strategy(user_input="", market_conditions=market_conditions)
        strategy_type = getattr(strategy, 'strategy_type', None)
        entry_conditions = getattr(strategy, 'entry_conditions', [])
        exit_conditions = getattr(strategy, 'exit_conditions', [])
        risk_parameters = getattr(strategy, 'risk_parameters', None)
        핵심지표 = {"전략": strategy_type.value if strategy_type else None, "진입조건": [ec.indicator for ec in entry_conditions], "청산조건": [ec.indicator for ec in exit_conditions], "리스크파라미터": risk_parameters.dict() if risk_parameters else {}}
        주장 = f"전략: {strategy_type.value if strategy_type else None}, 진입조건: {[ec.indicator for ec in entry_conditions]}, 청산조건: {[ec.indicator for ec in exit_conditions]}, 리스크파라미터: {risk_parameters.dict() if risk_parameters else {}}. "
        # 전략별 추천 및 신뢰도 예시
        if strategy_type and strategy_type.value.lower() == 'momentum':
            추천 = 'BUY'
            신뢰도 = 0.8
            주장 += '모멘텀 전략상 매수 우위.'
        elif strategy_type and strategy_type.value.lower() == 'mean_reversion':
            추천 = 'SELL'
            신뢰도 = 0.7
            주장 += '역추세 전략상 매도 우위.'
        else:
            추천 = 'HOLD'
            신뢰도 = 0.5
            주장 += '전략상 중립.'
        prompt = f"""너는 투자 전략 전문가야. 아래 전략 정보를 바탕으로 투자자에게 논리적으로 설명해줘.\n전략: {strategy_type.value if strategy_type else None}, 진입조건: {[ec.indicator for ec in entry_conditions]}, 청산조건: {[ec.indicator for ec in exit_conditions]}, 리스크파라미터: {risk_parameters.dict() if risk_parameters else {}}\n이 전략이 의미하는 바와 투자 판단에 미치는 영향, 추천 의견을 전문가답게 3~4문장으로 써줘."""
        전문가설명 = call_openai_api(prompt)
        return {
            "agent": "strategy_generator",
            "분야": "전략",
            "핵심지표": 핵심지표,
            "주장": 주장,
            "추천": 추천,
            "신뢰도": 신뢰도,
            "전문가설명": 전문가설명
        } 