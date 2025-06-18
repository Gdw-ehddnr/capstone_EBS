import numpy as np
import pandas as pd
import yfinance as yf
import investpy
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import talib
from .models import (
    TimeFrame, SignalStrength, TrendDirection,
    MarketData, MovingAverages, Oscillators,
    VolumeIndicators, SupportResistance, PatternSignal,
    TechnicalIndicators, TradingSignal,
    TechnicalAnalysisRequest, TechnicalAnalysisResponse,
    MovingAverageRequest, SupportResistanceRequest,
    PatternRequest, SignalRequest,
    MovingAverageResponse, PatternResponse,
    SignalResponse, Pattern, Signal
)
import logging
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import warnings
import time
from agents.utils.call_openai_api import call_openai_api
warnings.filterwarnings('ignore')

class TechnicalAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # 로그 포맷 설정
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # 파일 핸들러 추가
        fh = logging.FileHandler(f'logs/technical_analyzer_{datetime.now().strftime("%Y%m%d")}.log')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        # 콘솔 핸들러 추가
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        # LSTM 모델 초기화
        self.lstm_model = None
        self.scaler = MinMaxScaler()
        
        self.timeframe_map = {
            TimeFrame.M1: "1m",
            TimeFrame.M5: "5m",
            TimeFrame.M15: "15m",
            TimeFrame.M30: "30m",
            TimeFrame.H1: "1h",
            TimeFrame.H4: "4h",
            TimeFrame.D1: "1d",
            TimeFrame.W1: "1w",
            TimeFrame.MN1: "1mo"
        }
        self.indicators = {
            "sma": self._calculate_sma,
            "ema": self._calculate_ema,
            "rsi": self._calculate_rsi,
            "macd": self._calculate_macd
        }
        self.pattern_detectors = {
            "double_top": self._detect_double_top,
            "double_bottom": self._detect_double_bottom,
            "head_shoulders": self._detect_head_shoulders
        }
        self.signal_generators = {
            "trend_following": self._generate_trend_signal,
            "mean_reversion": self._generate_reversion_signal,
            "breakout": self._generate_breakout_signal
        }
        
        self.timeframe = TimeFrame.D1  # 기본값으로 일봉 설정
        self.patterns = {}
        self.trends = {}
        self.volume_analysis = {}
        self.forecast = {}
        self.analysis_cache = {}
        
    def analyze(self, symbol: str, period: str = "1y") -> Dict:
        """
        기술적 분석을 수행합니다.
        """
        try:
            self.logger.info(f"{symbol} 기술적 분석 시작")
            
            # 1. 데이터 수집
            self.logger.info(f"{symbol} 데이터 수집 시작")
            data = self._fetch_historical_data(symbol, period)
            self.logger.info(f"{symbol} 데이터 수집 완료: {len(data)} rows")
            
            # 방어 코드 추가: 데이터가 비어 있거나 'Close' 컬럼이 없으면 바로 리턴
            if data.empty or 'Close' not in data.columns:
                self.logger.warning(f"{symbol}의 데이터가 비어 있거나 'Close' 컬럼이 없습니다.")
                return None
            
            self.logger.info(f"최근 종가: {data['Close'].iloc[-1]:,.0f}원")
            
            # 2. 기술적 지표 계산
            self.logger.info(f"{symbol} 기술적 지표 계산 시작")
            indicators = self._calculate_technical_indicators(data)
            self.logger.info(f"RSI(14): {indicators['rsi'].iloc[-1]:.2f}")
            self.logger.info(f"MACD: {indicators['macd'].iloc[-1]:.2f}")
            self.logger.info(f"MACD Signal: {indicators['macd_signal'].iloc[-1]:.2f}")
            self.logger.info(f"볼린저 밴드 상단: {indicators['bb_upper'].iloc[-1]:,.0f}")
            self.logger.info(f"볼린저 밴드 중간: {indicators['bb_middle'].iloc[-1]:,.0f}")
            self.logger.info(f"볼린저 밴드 하단: {indicators['bb_lower'].iloc[-1]:,.0f}")
            
            # 3. 패턴 분석
            self.logger.info(f"{symbol} 패턴 분석 시작")
            patterns = self._analyze_chart_patterns(data)
            for pattern in patterns:
                self.logger.info(f"감지된 패턴: {pattern['pattern']}, 방향: {pattern['direction']}, 강도: {pattern['strength']}")
            
            # 4. 추세 분석
            self.logger.info(f"{symbol} 추세 분석 시작")
            trend_analysis = self._analyze_trend(data)
            self.logger.info(f"단기 추세: {trend_analysis['short_term_trend']}")
            self.logger.info(f"중기 추세: {trend_analysis['mid_term_trend']}")
            self.logger.info(f"장기 추세: {trend_analysis['long_term_trend']}")
            self.logger.info(f"추세 강도: {trend_analysis['trend_strength']:.2f}")
            
            # 5. 거래량 분석
            self.logger.info(f"{symbol} 거래량 분석 시작")
            volume_analysis = self._analyze_volume(data)
            self.logger.info(f"거래량 추세: {volume_analysis['volume_trend']}")
            self.logger.info(f"거래량 비율: {volume_analysis['volume_ratio']:.2f}")
            self.logger.info(f"VWAP: {volume_analysis['vwap']:,.0f}")
            
            # 6. 시계열 예측
            self.logger.info(f"{symbol} 시계열 예측 시작")
            price_forecast = self._forecast_price(data)
            for i, (date, price) in enumerate(zip(price_forecast['forecast_dates'], price_forecast['forecast_prices'])):
                self.logger.info(f"{i+1}일 후 예상가격: {price:,.0f}원")
            
            # 7. 분석 결과 종합
            self.logger.info(f"{symbol} 분석 결과 종합 시작")
            analysis = {
                'symbol': symbol,
                'timeframe': self.timeframe_map[self.timeframe],
                'current_price': data['Close'].iloc[-1],
                'indicators': indicators,
                'patterns': patterns,
                'trend_analysis': trend_analysis,
                'volume_analysis': volume_analysis,
                'price_forecast': price_forecast,
                'analysis_summary': self._generate_analysis_summary(
                    indicators, patterns, trend_analysis, volume_analysis, price_forecast
                ),
                'timestamp': datetime.now().isoformat(),
                'raw_data': data  # 실제 시계열 데이터 포함
            }
            
            # 8. 매매 신호 출력
            signals = self._generate_trading_signals(indicators, patterns, trend_analysis)
            for signal in signals:
                self.logger.info(f"매매 신호: {signal['type']}, 지표: {signal['indicator']}, 강도: {signal['strength']}, 이유: {signal['reason']}")
            
            self.logger.info(f"{symbol} 분석 결과 종합 완료")
            return analysis
            
        except Exception as e:
            self.logger.error(f"기술적 분석 중 오류 발생: {str(e)}")
            return None

    def analyze_technical_indicators(self, symbol: str, period: str = "1y") -> Dict:
        """
        기술적 지표를 분석합니다.
        """
        try:
            # 1. 데이터 수집
            data = self._fetch_historical_data(symbol, period)
            if data is None or len(data) == 0:
                raise ValueError("데이터가 비어 있습니다.")
            # 2. 기본 기술적 지표 계산
            indicators = self._calculate_technical_indicators(data)
            # 3. 차트 패턴 분석
            patterns = self._analyze_chart_patterns(data)
            # 4. 추세 분석
            trend_analysis = self._analyze_trend(data)
            # 5. 거래량 분석
            volume_analysis = self._analyze_volume(data)
            # 6. 시계열 예측
            price_forecast = self._forecast_price(data)
            return {
                'symbol': symbol,
                'technical_indicators': indicators,
                'chart_patterns': patterns,
                'trend_analysis': trend_analysis,
                'volume_analysis': volume_analysis,
                'price_forecast': price_forecast,
                'analysis_summary': self._generate_analysis_summary(
                    indicators, patterns, trend_analysis, volume_analysis, price_forecast
                )
            }
        except Exception as e:
            self.logger.error(f"기술적 분석 중 오류 발생: {str(e)}")
            return {
                'symbol': symbol,
                'technical_indicators': {},
                'chart_patterns': [],
                'trend_analysis': {},
                'volume_analysis': {},
                'price_forecast': {},
                'analysis_summary': f"기술적 분석 중 오류 발생: {str(e)}"
            }

    def _fetch_historical_data(self, symbol: str, period: str) -> pd.DataFrame:
        """
        국내 주식(코스피/코스닥)은 investpy만 사용, investpy 실패 시 빈 DataFrame 반환. 해외 주식만 yfinance 사용. 한글 종목명도 investpy로 강제 처리.
        """
        # 한글 종목명 → 코드 변환
        if not symbol.isdigit() and not symbol.endswith('.KS') and not symbol.endswith('.KQ'):
            code = self._get_stock_code(symbol) if hasattr(self, '_get_stock_code') else None
            if code:
                symbol = code
            else:
                self.logger.warning(f"한글 종목명 {symbol}의 코드 변환에 실패했습니다. 데이터 수집 불가.")
                return pd.DataFrame()
        is_kor = symbol.isdigit() or symbol.endswith('.KS') or symbol.endswith('.KQ')
        try:
            max_retries = 5
            for attempt in range(max_retries + 1):
                try:
                    if is_kor:
                        from datetime import datetime, timedelta
                        today = datetime.today()
                        if period.endswith('y'):
                            years = int(period[:-1])
                            start = today - timedelta(days=365*years)
                        elif period.endswith('m'):
                            months = int(period[:-1])
                            start = today - timedelta(days=30*months)
                        elif period.endswith('d'):
                            days = int(period[:-1])
                            start = today - timedelta(days=days)
                        else:
                            start = today - timedelta(days=365)
                        from_date = start.strftime('%d/%m/%Y')
                        to_date = today.strftime('%d/%m/%Y')
                        code = symbol.replace('.KS','').replace('.KQ','')
                        import investpy
                        data = investpy.get_stock_historical_data(stock=code, country='south korea', from_date=from_date, to_date=to_date)
                        if not data.empty:
                            data.index = pd.to_datetime(data.index)
                            return data
                        else:
                            self.logger.warning(f"investpy로 {symbol} 데이터가 비어 있습니다.")
                            return pd.DataFrame()
                    else:
                        import yfinance as yf
                        data = yf.download(symbol, period=period)
                        if data.empty:
                            self.logger.warning(f"yfinance로 {symbol} 데이터가 비어 있습니다.")
                            return pd.DataFrame()
                        return data
                except Exception as e:
                    err_msg = str(e)
                    if ("403" in err_msg or "ERR#0015" in err_msg) and attempt < max_retries:
                        self.logger.warning(f"{symbol} 데이터 수집 403 에러 발생, {attempt+1}회 재시도...")
                        time.sleep(2)
                        continue
                    self.logger.error(f"데이터 수집 실패: {err_msg}")
                    return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"데이터 수집 실패: {str(e)}")
            return pd.DataFrame()

    def _calculate_technical_indicators(self, data: pd.DataFrame) -> Dict:
        """
        기술적 지표를 계산합니다.
        """
        indicators = {}
        
        # 이동평균선
        indicators['sma_20'] = talib.SMA(data['Close'], timeperiod=20)
        indicators['sma_50'] = talib.SMA(data['Close'], timeperiod=50)
        indicators['sma_200'] = talib.SMA(data['Close'], timeperiod=200)
        
        # RSI
        indicators['rsi'] = talib.RSI(data['Close'], timeperiod=14)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(
            data['Close'], 
            fastperiod=12, 
            slowperiod=26, 
            signalperiod=9
        )
        indicators['macd'] = macd
        indicators['macd_signal'] = macd_signal
        indicators['macd_hist'] = macd_hist
        
        # 볼린저 밴드
        upper, middle, lower = talib.BBANDS(
            data['Close'],
            timeperiod=20,
            nbdevup=2,
            nbdevdn=2
        )
        indicators['bb_upper'] = upper
        indicators['bb_middle'] = middle
        indicators['bb_lower'] = lower
        
        # 스토캐스틱
        slowk, slowd = talib.STOCH(
            data['High'],
            data['Low'],
            data['Close'],
            fastk_period=14,
            slowk_period=3,
            slowk_matype=0,
            slowd_period=3,
            slowd_matype=0
        )
        indicators['stoch_k'] = slowk
        indicators['stoch_d'] = slowd
        
        return indicators

    def _analyze_chart_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """
        차트 패턴을 분석합니다.
        """
        patterns = []
        # 데이터가 충분한지 확인
        if data is None or len(data) < 7:
            return patterns
        
        # 헤드앤숄더 패턴 (임시: Harami를 사용)
        head_shoulders = talib.CDLHARAMI(data['Open'], data['High'], data['Low'], data['Close'])
        if head_shoulders.iloc[-1] != 0:
            patterns.append({
                'pattern': 'HEAD_AND_SHOULDERS',
                'direction': 'BEARISH' if head_shoulders.iloc[-1] < 0 else 'BULLISH',
                'strength': abs(head_shoulders.iloc[-1])
            })
        
        # 더블탑/더블바텀: 간단한 로직으로 대체
        close = data['Close']
        if len(close) >= 5:
            # 더블탑: 최근 5개 중 2개의 고점이 비슷하고, 그 사이에 저점이 있으면 감지
            peaks = (close[-5] < close[-4] > close[-3]) and (close[-1] < close[-2] > close[-3])
            if peaks and abs(close[-4] - close[-2]) < 1.0:  # 고점 차이 임계값
                patterns.append({
                    'pattern': 'DOUBLE_TOP',
                    'direction': 'BEARISH',
                    'strength': 1
                })
            # 더블바텀: 최근 5개 중 2개의 저점이 비슷하고, 그 사이에 고점이 있으면 감지
            troughs = (close[-5] > close[-4] < close[-3]) and (close[-1] > close[-2] < close[-3])
            if troughs and abs(close[-4] - close[-2]) < 1.0:
                patterns.append({
                    'pattern': 'DOUBLE_BOTTOM',
                    'direction': 'BULLISH',
                    'strength': 1
                })
        
        # 삼각형 패턴 등은 추후 구현
        return patterns

    def _analyze_trend(self, data: pd.DataFrame) -> Dict:
        """
        추세를 분석합니다.
        """
        if data is None or len(data) < 200:
            return {
                'short_term_trend': 'DOWN',
                'mid_term_trend': 'DOWN',
                'long_term_trend': 'DOWN',
                'trend_strength': 0
            }
        # 단기 추세 (20일)
        short_trend = 'UP' if data['Close'].iloc[-1] > data['Close'].iloc[-20] else 'DOWN'
        # 중기 추세 (50일)
        mid_trend = 'UP' if data['Close'].iloc[-1] > data['Close'].iloc[-50] else 'DOWN'
        # 장기 추세 (200일)
        long_trend = 'UP' if data['Close'].iloc[-1] > data['Close'].iloc[-200] else 'DOWN'
        # 추세 강도 계산
        trend_strength = self._calculate_trend_strength(data)
        return {
            'short_term_trend': short_trend,
            'mid_term_trend': mid_trend,
            'long_term_trend': long_trend,
            'trend_strength': trend_strength
        }

    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """
        추세의 강도를 계산합니다.
        """
        # ADX 지표 사용
        adx = talib.ADX(data['High'], data['Low'], data['Close'], timeperiod=14)
        return adx.iloc[-1]

    def _analyze_volume(self, data: pd.DataFrame) -> Dict:
        """
        거래량을 분석합니다.
        """
        # 거래량 이동평균
        volume_sma = talib.SMA(data['Volume'], timeperiod=20)
        
        # 거래량 추세
        volume_trend = 'UP' if data['Volume'].iloc[-1] > volume_sma.iloc[-1] else 'DOWN'
        
        # 거래량 가중 가격
        vwap = self._calculate_vwap(data)
        
        return {
            'volume_trend': volume_trend,
            'volume_sma': volume_sma.iloc[-1],
            'current_volume': data['Volume'].iloc[-1],
            'volume_ratio': data['Volume'].iloc[-1] / volume_sma.iloc[-1],
            'vwap': vwap
        }

    def _calculate_vwap(self, data: pd.DataFrame) -> float:
        """
        VWAP(Volume Weighted Average Price)를 계산합니다.
        """
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        vwap = (typical_price * data['Volume']).sum() / data['Volume'].sum()
        return vwap

    def _prepare_lstm_data(self, data: pd.DataFrame, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """
        LSTM 모델을 위한 데이터를 준비합니다.
        """
        # 종가 데이터만 사용
        prices = data['Close'].values.reshape(-1, 1)
        
        # 데이터 정규화
        scaled_prices = self.scaler.fit_transform(prices)
        
        X, y = [], []
        for i in range(sequence_length, len(scaled_prices)):
            X.append(scaled_prices[i-sequence_length:i, 0])
            y.append(scaled_prices[i, 0])
            
        return np.array(X), np.array(y)

    def _build_lstm_model(self, sequence_length: int) -> Sequential:
        """
        LSTM 모델을 구축합니다.
        """
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(sequence_length, 1)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def _forecast_price(self, data: pd.DataFrame, forecast_days: int = 5) -> Dict:
        """
        가격을 예측합니다.
        """
        try:
            # LSTM 데이터 준비
            X, y = self._prepare_lstm_data(data)
            
            # 모델 구축 및 학습
            if self.lstm_model is None:
                self.lstm_model = self._build_lstm_model(sequence_length=60)
                self.lstm_model.fit(X, y, epochs=50, batch_size=32, verbose=0)
            
            # 예측을 위한 최근 데이터 준비
            last_sequence = data['Close'].values[-60:].reshape(-1, 1)
            last_sequence = self.scaler.transform(last_sequence)
            
            # 예측
            forecast = []
            current_sequence = last_sequence.copy()
            
            for _ in range(forecast_days):
                next_pred = self.lstm_model.predict(current_sequence.reshape(1, 60, 1), verbose=0)
                forecast.append(next_pred[0, 0])
                current_sequence = np.roll(current_sequence, -1)
                current_sequence[-1] = next_pred
            
            # 예측값 역정규화
            forecast = self.scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
            
            return {
                'forecast_prices': forecast.flatten().tolist(),
                'forecast_dates': pd.date_range(
                    start=data.index[-1] + timedelta(days=1),
                    periods=forecast_days
                ).tolist()
            }
            
        except Exception as e:
            self.logger.error(f"가격 예측 중 오류 발생: {str(e)}")
            return {
                'forecast_prices': [],
                'forecast_dates': []
            }

    def _generate_analysis_summary(
        self,
        indicators: Dict,
        patterns: List[Dict],
        trend_analysis: Dict,
        volume_analysis: Dict,
        price_forecast: Dict
    ) -> Dict:
        """
        분석 결과를 종합하여 요약합니다.
        """
        # 매수/매도 신호 생성
        signals = self._generate_trading_signals(indicators, patterns, trend_analysis)
        
        # 신뢰도 점수 계산
        confidence_score = self._calculate_confidence_score(
            indicators, patterns, trend_analysis, volume_analysis
        )
        
        return {
            'trading_signals': signals,
            'confidence_score': confidence_score,
            'key_findings': self._summarize_key_findings(
                indicators, patterns, trend_analysis, volume_analysis
            )
        }

    def _generate_trading_signals(self, indicators: Dict, patterns: List[Dict], trend_analysis: Dict) -> List[Dict]:
        signals = []
        # RSI 기반 신호
        if 'rsi' in indicators and hasattr(indicators['rsi'], 'iloc') and not indicators['rsi'].isna().all():
            if indicators['rsi'].iloc[-1] < 30:
                signals.append({
                    'type': 'BUY',
                    'indicator': 'RSI',
                    'strength': 'STRONG',
                    'reason': '과매도 상태'
                })
            elif indicators['rsi'].iloc[-1] > 70:
                signals.append({
                    'type': 'SELL',
                    'indicator': 'RSI',
                    'strength': 'STRONG',
                    'reason': '과매수 상태'
                })
        # MACD 기반 신호
        if 'macd' in indicators and 'macd_signal' in indicators and hasattr(indicators['macd'], 'iloc') and not indicators['macd'].isna().all():
            if indicators['macd'].iloc[-1] > indicators['macd_signal'].iloc[-1] and \
               indicators['macd'].iloc[-2] <= indicators['macd_signal'].iloc[-2]:
                signals.append({
                    'type': 'BUY',
                    'indicator': 'MACD',
                    'strength': 'MODERATE',
                    'reason': 'MACD 골든크로스'
                })
        # 추세 기반 신호
        if trend_analysis.get('short_term_trend') == 'UP' and \
           trend_analysis.get('mid_term_trend') == 'UP' and \
           trend_analysis.get('trend_strength', 0) > 25:
            signals.append({
                'type': 'BUY',
                'indicator': 'TREND',
                'strength': 'STRONG',
                'reason': '강한 상승 추세'
            })
        # 패턴 기반 신호 예시 (옵션)
        for pattern in patterns:
            if pattern['pattern'] == 'DOUBLE_TOP':
                signals.append({
                    'type': 'SELL',
                    'indicator': 'PATTERN',
                    'strength': 'MODERATE',
                    'reason': '더블탑 패턴 감지'
                })
            elif pattern['pattern'] == 'DOUBLE_BOTTOM':
                signals.append({
                    'type': 'BUY',
                    'indicator': 'PATTERN',
                    'strength': 'MODERATE',
                    'reason': '더블바텀 패턴 감지'
                })
        return signals

    def _calculate_confidence_score(
        self,
        indicators: Dict,
        patterns: List[Dict],
        trend_analysis: Dict,
        volume_analysis: Dict
    ) -> float:
        """
        분석의 신뢰도 점수를 계산합니다.
        """
        score = 0.0
        total_factors = 0
        
        # RSI 신뢰도
        if 30 <= indicators['rsi'].iloc[-1] <= 70:
            score += 0.8
        total_factors += 1
        
        # MACD 신뢰도
        if abs(indicators['macd'].iloc[-1] - indicators['macd_signal'].iloc[-1]) > \
           abs(indicators['macd'].iloc[-2] - indicators['macd_signal'].iloc[-2]):
            score += 0.7
        total_factors += 1
        
        # 추세 신뢰도
        if trend_analysis['trend_strength'] > 25:
            score += 0.9
        total_factors += 1
        
        # 거래량 신뢰도
        if volume_analysis['volume_ratio'] > 1.5:
            score += 0.8
        total_factors += 1
        
        return score / total_factors

    def _summarize_key_findings(
        self,
        indicators: Dict,
        patterns: List[Dict],
        trend_analysis: Dict,
        volume_analysis: Dict
    ) -> List[str]:
        """
        주요 분석 결과를 요약합니다.
        """
        findings = []
        
        # RSI 상태
        rsi = indicators['rsi'].iloc[-1]
        if rsi < 30:
            findings.append("RSI가 과매도 상태입니다.")
        elif rsi > 70:
            findings.append("RSI가 과매수 상태입니다.")
        
        # 추세 상태
        if trend_analysis['trend_strength'] > 25:
            findings.append(f"강한 {trend_analysis['short_term_trend']} 추세가 형성되어 있습니다.")
        
        # 거래량 상태
        if volume_analysis['volume_ratio'] > 1.5:
            findings.append("거래량이 평균보다 크게 증가했습니다.")
        
        # 차트 패턴
        for pattern in patterns:
            findings.append(f"{pattern['pattern']} 패턴이 감지되었습니다.")
        
        return findings

    def _fetch_market_data(
        self,
        ticker: str,
        timeframe: TimeFrame,
        lookback_periods: int
    ) -> List[MarketData]:
        """시장 데이터 조회"""
        # yfinance를 통한 데이터 조회
        yf_timeframe = self.timeframe_map[timeframe]
        stock = yf.Ticker(ticker)
        
        # 데이터 기간 계산
        if timeframe in [TimeFrame.M1, TimeFrame.M5, TimeFrame.M15, TimeFrame.M30]:
            period = "7d"  # yfinance 제한
        else:
            period = f"{lookback_periods * 2}d"
            
        df = stock.history(period=period, interval=yf_timeframe)
        
        # MarketData 객체 리스트 생성
        market_data = []
        for index, row in df.iterrows():
            market_data.append(MarketData(
                ticker=ticker,
                timestamp=index,
                open=row['Open'],
                high=row['High'],
                low=row['Low'],
                close=row['Close'],
                volume=row['Volume'],
                adjusted_close=row.get('Adj Close', None)
            ))
            
        return market_data[-lookback_periods:]
    
    def _convert_nan_to_none(self, value):
        """NaN 값을 None으로 변환"""
        return None if pd.isna(value) else float(value)

    def _calculate_moving_averages(self, request: MovingAverageRequest) -> MovingAverageResponse:
        """이동평균 계산"""
        try:
            # Convert market data to numpy array
            prices = np.array([data.close for data in request.market_data])
            
            # Check if data length is sufficient
            if len(prices) < max(request.window_sizes):
                raise ValueError("데이터 길이가 충분하지 않습니다.")
            
            sma_values = {}
            ema_values = {}
            
            for window in request.window_sizes:
                if "sma" in request.ma_types:
                    sma = self._calculate_sma(prices, window)
                    sma_values[f"sma_{window}"] = float(sma[-1])
                
                if "ema" in request.ma_types:
                    ema = self._calculate_ema(prices, window)
                    ema_values[f"ema_{window}"] = float(ema[-1])
            
            trend = self._determine_trend_direction(prices)
            
            return MovingAverageResponse(
                sma_values=sma_values,
                ema_values=ema_values,
                trend_direction=trend,
                error_message=None
            )
        except Exception as e:
            return MovingAverageResponse(
                sma_values={},
                ema_values={},
                trend_direction=TrendDirection.SIDEWAYS,
                error_message=str(e)
            )

    def _calculate_oscillators(self, df: pd.DataFrame) -> Oscillators:
        """오실레이터 계산"""
        # Check if data length is sufficient
        if len(df) < 14:
            raise ValueError("데이터 길이가 충분하지 않습니다.")
        
        # RSI
        rsi_14 = self._convert_nan_to_none(talib.RSI(df['Close'], timeperiod=14)[-1])
        
        # Stochastic
        stoch_k, stoch_d = talib.STOCH(
            df['High'],
            df['Low'],
            df['Close'],
            fastk_period=14,
            slowk_period=3,
            slowd_period=3
        )
        
        # MACD
        macd_line, macd_signal, macd_hist = talib.MACD(
            df['Close'],
            fastperiod=12,
            slowperiod=26,
            signalperiod=9
        )
        
        # CCI
        cci_20 = self._convert_nan_to_none(talib.CCI(
            df['High'],
            df['Low'],
            df['Close'],
            timeperiod=20
        )[-1])
        
        # ATR
        atr_14 = self._convert_nan_to_none(talib.ATR(
            df['High'],
            df['Low'],
            df['Close'],
            timeperiod=14
        )[-1])
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(
            df['Close'],
            timeperiod=20,
            nbdevup=2,
            nbdevdn=2,
            matype=0
        )
        
        return Oscillators(
            rsi_14=rsi_14,
            stoch_k=self._convert_nan_to_none(stoch_k[-1]),
            stoch_d=self._convert_nan_to_none(stoch_d[-1]),
            macd_line=self._convert_nan_to_none(macd_line[-1]),
            macd_signal=self._convert_nan_to_none(macd_signal[-1]),
            macd_histogram=self._convert_nan_to_none(macd_hist[-1]),
            cci_20=cci_20,
            atr_14=atr_14,
            bollinger_upper=self._convert_nan_to_none(upper[-1]),
            bollinger_middle=self._convert_nan_to_none(middle[-1]),
            bollinger_lower=self._convert_nan_to_none(lower[-1])
        )
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> VolumeIndicators:
        """거래량 지표 계산"""
        # Check if data length is sufficient
        if len(df) < 20:
            raise ValueError("데이터 길이가 충분하지 않습니다.")
        
        # On Balance Volume
        obv = self._convert_nan_to_none(talib.OBV(df['Close'], df['Volume'])[-1])
        
        # Volume SMA
        volume_sma_20 = self._convert_nan_to_none(talib.SMA(df['Volume'], timeperiod=20)[-1])
        
        # Volume Trend
        recent_volume = df['Volume'].tail(5).mean()
        volume_trend = TrendDirection.SIDEWAYS
        
        if volume_sma_20 is not None and not pd.isna(recent_volume):
            if recent_volume > volume_sma_20 * 1.5:
                volume_trend = TrendDirection.STRONG_UPTREND
            elif recent_volume > volume_sma_20:
                volume_trend = TrendDirection.UPTREND
            elif recent_volume < volume_sma_20 * 0.5:
                volume_trend = TrendDirection.STRONG_DOWNTREND
            elif recent_volume < volume_sma_20:
                volume_trend = TrendDirection.DOWNTREND
            
        # Price Volume Trend
        pvt = (df['Close'].diff() / df['Close'].shift(1)) * df['Volume']
        pvt = pvt.cumsum()
        
        # Money Flow Index
        mfi = self._convert_nan_to_none(talib.MFI(
            df['High'],
            df['Low'],
            df['Close'],
            df['Volume'],
            timeperiod=14
        )[-1])
        
        return VolumeIndicators(
            obv=obv,
            volume_sma_20=volume_sma_20,
            volume_trend=volume_trend,
            volume_price_trend=self._convert_nan_to_none(pvt.iloc[-1]),
            money_flow_index=mfi
        )
    
    def _calculate_support_resistance(
        self,
        df: pd.DataFrame,
        n_levels: int = 3
    ) -> SupportResistance:
        """지지/저항 레벨 계산"""
        # 피봇 포인트 계산
        pivot = (df['High'].iloc[-1] + df['Low'].iloc[-1] + df['Close'].iloc[-1]) / 3
        
        # 지지/저항 레벨
        r1 = 2 * pivot - df['Low'].iloc[-1]
        r2 = pivot + (df['High'].iloc[-1] - df['Low'].iloc[-1])
        s1 = 2 * pivot - df['High'].iloc[-1]
        s2 = pivot - (df['High'].iloc[-1] - df['Low'].iloc[-1])
        
        # 추가 지지/저항 레벨 탐색
        highs = df['High'].nlargest(10)
        lows = df['Low'].nsmallest(10)
        
        resistance_levels = list(set([round(x, 2) for x in highs.values]))[:n_levels]
        support_levels = list(set([round(x, 2) for x in lows.values]))[:n_levels]
        
        return SupportResistance(
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            pivot_point=pivot,
            r1_level=r1,
            r2_level=r2,
            s1_level=s1,
            s2_level=s2
        )
    
    def _detect_patterns(self, df: pd.DataFrame) -> List[PatternSignal]:
        """차트 패턴 탐지"""
        patterns = []
        
        # Candlestick Patterns
        doji = talib.CDLDOJI(
            df['Open'],
            df['High'],
            df['Low'],
            df['Close']
        )
        hammer = talib.CDLHAMMER(
            df['Open'],
            df['High'],
            df['Low'],
            df['Close']
        )
        engulfing = talib.CDLENGULFING(
            df['Open'],
            df['High'],
            df['Low'],
            df['Close']
        )
        
        # 최근 패턴 확인
        if doji.iloc[-1] != 0:
            patterns.append(PatternSignal(
                pattern_name="Doji",
                confidence=0.6,
                signal=SignalStrength.NEUTRAL
            ))
            
        if hammer.iloc[-1] != 0:
            patterns.append(PatternSignal(
                pattern_name="Hammer",
                confidence=0.7,
                signal=SignalStrength.BUY if hammer.iloc[-1] > 0 else SignalStrength.SELL
            ))
            
        if engulfing.iloc[-1] != 0:
            patterns.append(PatternSignal(
                pattern_name="Engulfing",
                confidence=0.8,
                signal=SignalStrength.BUY if engulfing.iloc[-1] > 0 else SignalStrength.SELL
            ))
            
        return patterns
    
    def _determine_trend_direction(self, prices: np.ndarray) -> TrendDirection:
        if len(prices) < 2:
            return TrendDirection.SIDEWAYS
        
        # Calculate price change
        price_change = (prices[-1] - prices[0]) / prices[0]
        
        if price_change > 0.02:  # 2% threshold for uptrend
            return TrendDirection.UP
        elif price_change < -0.02:  # -2% threshold for downtrend
            return TrendDirection.DOWN
        return TrendDirection.SIDEWAYS

    def _calculate_sma(self, prices: np.ndarray, window: int) -> np.ndarray:
        return np.convolve(prices, np.ones(window)/window, mode='valid')

    def _calculate_ema(self, prices: np.ndarray, window: int) -> np.ndarray:
        alpha = 2 / (window + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        return ema

    def _calculate_rsi(self, df: pd.DataFrame, params: Dict) -> float:
        window = params.get("window", 14)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs.iloc[-1]))

    def _calculate_macd(self, df: pd.DataFrame, params: Dict) -> Dict[str, float]:
        fast = params.get("fast", 12)
        slow = params.get("slow", 26)
        signal = params.get("signal", 9)
        
        fast_ema = df["close"].ewm(span=fast).mean()
        slow_ema = df["close"].ewm(span=slow).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal).mean()
        
        return {
            "macd": macd_line.iloc[-1],
            "signal": signal_line.iloc[-1],
            "histogram": macd_line.iloc[-1] - signal_line.iloc[-1]
        }

    def _detect_double_top(self, prices: np.ndarray) -> Optional[Pattern]:
        """더블 탑 패턴 감지"""
        if len(prices) < 5:
            return None

        # 두 개의 고점과 그 사이의 저점 확인
        peaks = (prices[1] > prices[0]) and (prices[1] > prices[2])
        trough = (prices[2] < prices[1]) and (prices[2] < prices[3])
        second_peak = (prices[3] > prices[2]) and (prices[3] > prices[4])

        if peaks and trough and second_peak:
            return Pattern(
            type="double_top",
            confidence=0.8,
            timestamp=datetime.now().isoformat()
        )
        return None

    def _detect_double_bottom(self, prices: np.ndarray) -> Optional[Pattern]:
        """더블 바텀 패턴 감지"""
        if len(prices) < 5:
            return None

        # 두 개의 저점과 그 사이의 고점 확인
        troughs = (prices[1] < prices[0]) and (prices[1] < prices[2])
        peak = (prices[2] > prices[1]) and (prices[2] > prices[3])
        second_trough = (prices[3] < prices[2]) and (prices[3] < prices[4])

        if troughs and peak and second_trough:
            return Pattern(
            type="double_bottom",
            confidence=0.8,
            timestamp=datetime.now().isoformat()
        )
        return None

    def _detect_head_shoulders(self, prices: np.ndarray) -> Optional[Pattern]:
        """헤드 앤 숄더 패턴 감지"""
        if len(prices) < 7:
            return None

        # 헤드 앤 숄더 패턴 확인
        left_shoulder = (prices[1] > prices[0]) and (prices[1] > prices[2])
        head = (prices[3] > prices[1]) and (prices[3] > prices[5])
        right_shoulder = (prices[5] > prices[4]) and (prices[5] > prices[6])

        if left_shoulder and head and right_shoulder:
            return Pattern(
            type="head_shoulders",
            confidence=0.8,
            timestamp=datetime.now().isoformat()
        )
        return None

    def _generate_trend_signal(self, prices: np.ndarray) -> Signal:
        if len(prices) < 20:
            return Signal(signal="NEUTRAL", strength="WEAK")

        # 가격 변화율 기준 (최근 종가 vs 시작 종가)
        price_change = (prices[-1] - prices[0]) / prices[0]

        # RSI 계산 (14)
        close_series = pd.Series(prices)
        delta = close_series.diff()
        gain = delta.clip(lower=0).rolling(window=14).mean()
        loss = -delta.clip(upper=0).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_latest = rsi.iloc[-1]

        # 조건 조합
        if price_change > 0.02 and rsi_latest < 70:
            return Signal(signal="BUY", strength="STRONG")
        elif price_change < -0.02 and rsi_latest > 30:
            return Signal(signal="SELL", strength="STRONG")
        else:
            return Signal(signal="NEUTRAL", strength="WEAK")

    def _generate_reversion_signal(self, prices: np.ndarray) -> Signal:
        if len(prices) < 20:
            return Signal(signal="NEUTRAL", strength="WEAK")

        close_series = pd.Series(prices)
        sma_20 = close_series.rolling(window=20).mean()
        std_20 = close_series.rolling(window=20).std()

        upper_band = sma_20 + 2 * std_20
        lower_band = sma_20 - 2 * std_20

        if prices[-1] < lower_band.iloc[-1]:
            return Signal(signal="BUY", strength="MODERATE")
        elif prices[-1] > upper_band.iloc[-1]:
            return Signal(signal="SELL", strength="MODERATE")
        else:
            return Signal(signal="NEUTRAL", strength="WEAK")

    def _generate_breakout_signal(self, prices: np.ndarray) -> Signal:
        if len(prices) < 20:
            return Signal(signal="NEUTRAL", strength="WEAK")

        recent_high = np.max(prices[-20:-1])
        recent_low = np.min(prices[-20:-1])
        current_price = prices[-1]

        if current_price > recent_high:
            return Signal(signal="BUY", strength="STRONG") 
        elif current_price < recent_low:
            return Signal(signal="SELL", strength="STRONG")
        else:
            return Signal(signal="NEUTRAL", strength="WEAK") 

    def propose(self, context):
        """
        자신의 기술적 분석 결과를 의견으로 제시합니다.
        """
        symbol = context.get('symbol', '005930')
        period = context.get('period', '1y')
        timestamp = context.get('timestamp', None)
        cache_key = (symbol, period, timestamp)
        if cache_key in self.analysis_cache:
            analysis = self.analysis_cache[cache_key]
        else:
            analysis = self.analyze(symbol, period)
            self.analysis_cache[cache_key] = analysis

        decision = 'HOLD'
        confidence = 0.5
        reasons = []

        if analysis:
            # 1. RSI, MACD
            indicators = analysis.get('indicators', {})
            rsi = indicators.get('rsi')
            macd = indicators.get('macd')
            macd_signal = indicators.get('macd_signal')
            rsi_val = rsi.iloc[-1] if rsi is not None and hasattr(rsi, 'iloc') else None
            macd_val = macd.iloc[-1] if macd is not None and hasattr(macd, 'iloc') else None
            macd_signal_val = macd_signal.iloc[-1] if macd_signal is not None and hasattr(macd_signal, 'iloc') else None

            # 2. 추세
            trend = analysis.get('trend_analysis', {})
            short_trend = trend.get('short_term_trend')
            mid_trend = trend.get('mid_term_trend')
            long_trend = trend.get('long_term_trend')
            trend_strength = trend.get('trend_strength', 0)

            # 3. 거래량
            volume = analysis.get('volume_analysis', {})
            volume_trend = volume.get('volume_trend')
            volume_ratio = volume.get('volume_ratio')

            # 4. 시계열 예측
            forecast = analysis.get('price_forecast', {})
            forecast_prices = forecast.get('forecast_prices', [])
            current_price = analysis.get('current_price')
            price_trend = None
            if forecast_prices and current_price is not None:
                future_price = forecast_prices[-1]
                if future_price > current_price * 1.01:
                    price_trend = 'UP'
                elif future_price < current_price * 0.99:
                    price_trend = 'DOWN'
                else:
                    price_trend = 'SIDEWAYS'

            # 종합 판단 로직
            buy_score = 0
            sell_score = 0
            hold_score = 0

            # RSI
            if rsi_val is not None:
                if rsi_val < 30:
                    buy_score += 2
                    reasons.append(f'RSI 과매도({rsi_val:.2f})')
                elif rsi_val > 70:
                    sell_score += 2
                    reasons.append(f'RSI 과매수({rsi_val:.2f})')
                else:
                    hold_score += 1
                    reasons.append(f'RSI 중립({rsi_val:.2f})')

            # MACD
            if macd_val is not None and macd_signal_val is not None:
                if macd_val > macd_signal_val and macd_val > 0:
                    buy_score += 1
                    reasons.append(f'MACD 골든크로스({macd_val:.2f} > {macd_signal_val:.2f})')
                elif macd_val < macd_signal_val and macd_val < 0:
                    sell_score += 1
                    reasons.append(f'MACD 데드크로스({macd_val:.2f} < {macd_signal_val:.2f})')
                else:
                    hold_score += 1
                    reasons.append('MACD 중립')

            # 추세
            if short_trend == 'UP':
                buy_score += 1
                reasons.append('단기 추세 상승')
            elif short_trend == 'DOWN':
                sell_score += 1
                reasons.append('단기 추세 하락')
            if mid_trend == 'UP':
                buy_score += 1
                reasons.append('중기 추세 상승')
            elif mid_trend == 'DOWN':
                sell_score += 1
                reasons.append('중기 추세 하락')
            if long_trend == 'UP':
                buy_score += 1
                reasons.append('장기 추세 상승')
            elif long_trend == 'DOWN':
                sell_score += 1
                reasons.append('장기 추세 하락')
            if trend_strength > 25:
                reasons.append(f'추세 강도 강함({trend_strength:.2f})')
            else:
                reasons.append(f'추세 강도 약함({trend_strength:.2f})')

            # 거래량
            if volume_trend == 'UP' and volume_ratio is not None and volume_ratio > 1.2:
                buy_score += 1
                reasons.append(f'거래량 증가({volume_ratio:.2f}배)')
            elif volume_trend == 'DOWN' and volume_ratio is not None and volume_ratio < 0.8:
                sell_score += 1
                reasons.append(f'거래량 감소({volume_ratio:.2f}배)')
            else:
                hold_score += 1
                reasons.append('거래량 중립')

            # 시계열 예측
            if price_trend == 'UP':
                buy_score += 1
                reasons.append('시계열 예측: 상승')
            elif price_trend == 'DOWN':
                sell_score += 1
                reasons.append('시계열 예측: 하락')
            else:
                hold_score += 1
                reasons.append('시계열 예측: 횡보')

            # 최종 결정
            if buy_score > sell_score and buy_score > hold_score:
                decision = 'BUY'
                confidence = min(1.0, 0.5 + 0.1 * buy_score)
            elif sell_score > buy_score and sell_score > hold_score:
                decision = 'SELL'
                confidence = min(1.0, 0.5 + 0.1 * sell_score)
            else:
                decision = 'HOLD'
                confidence = 0.5

        reason = ', '.join(reasons) if reasons else '기술적 분석 결과'
        return {
            'agent': 'technical_analyzer',
            'decision': decision,
            'confidence': confidence,
            'reason': reason
        }

    def debate(self, context, others_opinions, my_opinion_1st_round=None):
        analysis = self.analyze(context['symbol'], context.get('period', '1y'))
        
        # 분석 결과가 None인 경우 기본값 반환
        if analysis is None:
            return {
                "agent": "technical_analyzer",
                "분야": "기술적분석",
                "핵심지표": {},
                "주장": "데이터 수집 실패로 인해 기술적 분석을 수행할 수 없습니다.",
                "추천": "HOLD",
                "신뢰도": 0.0,
                "전문가설명": "현재 데이터 수집에 문제가 있어 기술적 분석을 수행할 수 없습니다. 나중에 다시 시도해주세요."
            }
            
        indicators = analysis.get('indicators', {})
        rsi = indicators.get('rsi')
        macd = indicators.get('macd')
        macd_signal = indicators.get('macd_signal')
        rsi_val = rsi.iloc[-1] if rsi is not None and hasattr(rsi, 'iloc') else None
        macd_val = macd.iloc[-1] if macd is not None and hasattr(macd, 'iloc') else None
        macd_signal_val = macd_signal.iloc[-1] if macd_signal is not None and hasattr(macd_signal, 'iloc') else None
        trend = analysis.get('trend_analysis', {})
        short_trend = trend.get('short_term_trend')
        mid_trend = trend.get('mid_term_trend')
        long_trend = trend.get('long_term_trend')
        trend_strength = trend.get('trend_strength', 0)
        핵심지표 = {"RSI": rsi_val, "MACD": macd_val, "MACD시그널": macd_signal_val, "단기추세": short_trend, "중기추세": mid_trend, "장기추세": long_trend, "추세강도": trend_strength}
        주장 = f"RSI {rsi_val}, MACD {macd_val}, 단기추세 {short_trend}, 중기추세 {mid_trend}, 장기추세 {long_trend}, 추세강도 {trend_strength}. "
        buy_score = 0
        sell_score = 0
        hold_score = 0
        if rsi_val is not None:
            if rsi_val < 30:
                buy_score += 2
                주장 += f'RSI 과매도({rsi_val:.2f}). '
            elif rsi_val > 70:
                sell_score += 2
                주장 += f'RSI 과매수({rsi_val:.2f}). '
            else:
                hold_score += 1
                주장 += f'RSI 중립({rsi_val:.2f}). '
        if macd_val is not None and macd_signal_val is not None:
            if macd_val > macd_signal_val and macd_val > 0:
                buy_score += 1
                주장 += f'MACD 골든크로스({macd_val:.2f} > {macd_signal_val:.2f}). '
            elif macd_val < macd_signal_val and macd_val < 0:
                sell_score += 1
                주장 += f'MACD 데드크로스({macd_val:.2f} < {macd_signal_val:.2f}). '
            else:
                hold_score += 1
                주장 += 'MACD 중립. '
        if short_trend == 'UP':
            buy_score += 1
            주장 += '단기 추세 상승. '
        elif short_trend == 'DOWN':
            sell_score += 1
            주장 += '단기 추세 하락. '
        if mid_trend == 'UP':
            buy_score += 1
            주장 += '중기 추세 상승. '
        elif mid_trend == 'DOWN':
            sell_score += 1
            주장 += '중기 추세 하락. '
        if long_trend == 'UP':
            buy_score += 1
            주장 += '장기 추세 상승. '
        elif long_trend == 'DOWN':
            sell_score += 1
            주장 += '장기 추세 하락. '
        if trend_strength > 25:
            주장 += f'추세 강도 강함({trend_strength:.2f}). '
        else:
            주장 += f'추세 강도 약함({trend_strength:.2f}). '
        if buy_score > sell_score and buy_score > hold_score:
            추천 = 'BUY'
            신뢰도 = min(1.0, 0.5 + 0.1 * buy_score)
            주장 += '기술적 신호상 매수 우위.'
        elif sell_score > buy_score and sell_score > hold_score:
            추천 = 'SELL'
            신뢰도 = min(1.0, 0.5 + 0.1 * sell_score)
            주장 += '기술적 신호상 매도 우위.'
        else:
            추천 = 'HOLD'
            신뢰도 = 0.5
            주장 += '기술적 신호상 중립.'
        prompt = f"""너는 기술적 분석 전문가야. 아래 수치를 바탕으로 투자자에게 논리적으로 설명해줘.\nRSI: {rsi_val}, MACD: {macd_val}, 단기추세: {short_trend}, 중기추세: {mid_trend}, 장기추세: {long_trend}, 추세강도: {trend_strength}\n이 수치가 의미하는 바와 투자 판단에 미치는 영향, 추천 의견을 전문가답게 3~4문장으로 써줘."""
        전문가설명 = call_openai_api(prompt)
        return {
            "agent": "technical_analyzer",
            "분야": "기술적분석",
            "핵심지표": 핵심지표,
            "주장": 주장,
            "추천": 추천,
            "신뢰도": 신뢰도,
            "전문가설명": 전문가설명
        } 