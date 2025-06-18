import logging
from datetime import datetime
from typing import List, Dict, Optional
import asyncio
import pandas as pd
from agents.decision.decision_maker import DecisionMaker
from agents.news.news_analyzer import NewsAnalyzer
from agents.strategy.strategy_generator import StrategyGenerator, TimeHorizon
from agents.risk.risk_analyzer import RiskAnalyzer
from agents.technical.technical_analyzer import TechnicalAnalyzer
from agents.decision.models import (
    DecisionRequest, DecisionResponse, MarketContext,
    RiskParameters, AgentAnalysis, DecisionType
)
from agents.risk.models import RiskAssessmentRequest
from agents.strategy.models import MarketCondition
import FinanceDataReader as fdr
import openai
import os
import json
import csv

class SystemManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # 로그 포맷 설정
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # 파일 핸들러 추가
        fh = logging.FileHandler(f'logs/system_manager_{datetime.now().strftime("%Y%m%d")}.log')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        # 콘솔 핸들러 추가
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        # 에이전트 초기화
        self.news_analyzer = NewsAnalyzer()
        self.strategy_generator = StrategyGenerator()
        self.risk_analyzer = RiskAnalyzer()
        self.technical_analyzer = TechnicalAnalyzer()
        self.decision_maker = DecisionMaker()
        
        # 시스템 상태
        self.is_running = False
        self.current_symbol = None
        self.current_timeframe = None
        self.last_decision = None
        self.last_analysis_time = None
        # 종목명→코드 자동 매핑용 전체 종목 리스트 캐싱
        try:
            self.stock_df = fdr.StockListing('KRX')
            self.name_to_code = dict(zip(self.stock_df['Name'], self.stock_df['Code']))
        except Exception as e:
            self.logger.error(f"종목 리스트 로딩 실패: {str(e)}")
            self.stock_df = None
            self.name_to_code = {}

    async def start_system(self, symbol: str, timeframe: str):
        """
        시스템을 시작합니다.
        """
        try:
            self.logger.info(f"시스템 시작: {symbol} ({timeframe})")
            self.is_running = True
            self.current_symbol = symbol
            self.current_timeframe = timeframe
            
            # 에이전트 초기화
            await self._initialize_agents()
            
            # 주기적 분석 시작
            await self._start_periodic_analysis()
            
        except Exception as e:
            self.logger.error(f"시스템 시작 중 오류 발생: {str(e)}")
            raise

    async def stop_system(self):
        """
        시스템을 중지합니다.
        """
        try:
            self.logger.info("시스템 중지")
            self.is_running = False
        except Exception as e:
            self.logger.error(f"시스템 중지 중 오류 발생: {str(e)}")
            raise

    async def _initialize_agents(self):
        """
        모든 에이전트를 초기화합니다.
        """
        try:
            # 각 에이전트는 생성자에서 초기화를 수행하므로 별도의 initialize 호출이 필요하지 않음
            self.logger.info("모든 에이전트 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"에이전트 초기화 중 오류 발생: {str(e)}")
            raise

    def _create_risk_parameters(self, risk_analysis: Dict) -> RiskParameters:
        """
        리스크 파라미터를 생성합니다.
        """
        try:
            if isinstance(risk_analysis, dict):
                return RiskParameters(
                    max_position_size=0.1,  # 기본값
                    stop_loss_percentage=risk_analysis.get('stop_loss', 0.05),
                    take_profit_percentage=risk_analysis.get('target_price', 0.1),
                    max_drawdown=risk_analysis.get('max_loss', 0.1),
                    risk_reward_ratio=2.0  # 기본값
                )
            else:
                return RiskParameters(
                    max_position_size=0.1,
                    stop_loss_percentage=0.05,
                    take_profit_percentage=0.1,
                    max_drawdown=0.1,
                    risk_reward_ratio=2.0
                )
        except Exception as e:
            self.logger.error(f"리스크 파라미터 생성 중 오류 발생: {str(e)}")
            return RiskParameters(
                max_position_size=0.1,
                stop_loss_percentage=0.05,
                take_profit_percentage=0.1,
                max_drawdown=0.1,
                risk_reward_ratio=2.0
            )

    async def _start_periodic_analysis(self):
        """
        주기적인 분석을 시작합니다. (보유 종목 + 상위 10개 종목 분석)
        """
        while self.is_running:
            try:
                # 1. 보유 종목 불러오기
                holdings = self._load_holdings()
                holding_codes = list(holdings.keys())
                # 2. 상위 10개 종목 선정 (거래량 기준)
                if self.stock_df is not None:
                    top10_df = self.stock_df.sort_values(by='Volume', ascending=False).head(10)
                    top10_codes = list(top10_df['Code'])
                else:
                    self.logger.error("종목 데이터가 없습니다. 기본 종목 리스트로 대체합니다.")
                    top10_codes = [self.current_symbol]
                # 3. 보유 종목 + 상위 10개 종목 합치기(중복 제거, 최대 15개)
                all_codes = list(dict.fromkeys(holding_codes + top10_codes))[:15]
                # 분석 성공 종목만 append할 리스트
                news_analyses = []
                strategy_results = []
                technical_analyses = []
                risk_analyses = []
                debate_results_all = []
                target_symbols_filtered = []
                for symbol in all_codes:
                    self.logger.info(f"[{symbol}] 뉴스 분석 시작")
                    news = self.news_analyzer.analyze_sentiment_for_stock(symbol)
                    if not news or (isinstance(news, dict) and news.get('감성점수') is None):
                        self.logger.warning(f"[{symbol}] 뉴스 분석 실패 또는 결과 없음. 이후 분석 건너뜀.")
                        continue
                    self.logger.info(f"[{symbol}] 뉴스 분석 완료")
                    # MarketCondition 객체 변환
                    market_trend = 'neutral'
                    if isinstance(news, dict):
                        impact = news.get('시장영향도', 0.0)
                        if impact > 0:
                            market_trend = 'bullish'
                        elif impact < 0:
                            market_trend = 'bearish'
                    market_condition_obj = MarketCondition(
                        market_trend=market_trend,
                        volatility_level='high' if news.get('뉴스갯수', 0) >= 10 else 'low',
                        trading_volume=0.0,
                        sector_performance={},
                        major_events=[],
                        timestamp=datetime.now()
                    )
                    self.logger.info(f"[{symbol}] 전략 분석 시작")
                    strategy = self.strategy_generator.generate_strategy(
                        user_input="",
                        market_conditions=market_condition_obj,
                        risk_tolerance=0.5,
                        time_horizon=TimeHorizon.MEDIUM_TERM
                    )
                    self.logger.info(f"[{symbol}] 전략 분석 완료")
                    self.logger.info(f"[{symbol}] 기술적 분석 시작")
                    try:
                        tech = self.technical_analyzer.analyze(symbol, retry_count=5)
                    except TypeError:
                        tech = self.technical_analyzer.analyze(symbol)
                    if (
                        not tech
                        or not isinstance(tech, dict)
                        or 'raw_data' not in tech
                        or tech['raw_data'] is None
                        or not isinstance(tech['raw_data'], pd.DataFrame)
                        or tech['raw_data'].empty
                    ):
                        self.logger.warning(f"[{symbol}] 기술적 분석 실패 또는 데이터 없음. 이후 분석 건너뜀.")
                        continue
                    self.logger.info(f"[{symbol}] 기술적 분석 완료")
                    self.logger.info(f"[{symbol}] 리스크 분석 시작")
                    market_data = tech.get('raw_data', pd.DataFrame())
                    technical_indicators = tech.get('indicators', {})
                    risk = self.risk_analyzer.assess_risk(
                        request=RiskAssessmentRequest(
                            market_data=market_data,
                            technical_indicators=technical_indicators,
                            risk_tolerance=0.5,
                            portfolio_value=1000000.0,
                            positions=[],
                            market_context=None
                        )
                    )
                    if not risk:
                        self.logger.warning(f"[{symbol}] 리스크 분석 실패. 이후 분석 건너뜀.")
                        continue
                    self.logger.info(f"[{symbol}] 리스크 분석 완료")
                    news_analyses.append(news)
                    strategy_results.append(strategy)
                    technical_analyses.append(tech)
                    risk_analyses.append(risk)
                    target_symbols_filtered.append(symbol)
                    # [추가] debate 수행 및 결과 저장
                    debate_context = {
                        'symbol': symbol,
                        'market_conditions': market_condition_obj,
                        'period': '1y',
                        'timestamp': datetime.now().isoformat(),
                        'technical_indicators': tech.get('indicators', {}),
                        'market_data': tech.get('raw_data', None)
                    }
                    debate_opinions = self.agent_debate_round(debate_context)
                    debate_results_all.append(debate_opinions)
                # 기술적 분석 시계열 예측 CSV 저장
                self.save_forecast_csv(technical_analyses)
                # 이하 기존 top5 선정 및 debate/보고서/holdings 저장 로직 유지
                symbol_scores = []
                for i, debate_result in enumerate(debate_results_all):
                    buy_count = 0
                    confidence_sum = 0.0
                    # debate_result는 각 agent의 의견 리스트
                    for agent_opinion in debate_result:
                        if isinstance(agent_opinion, dict):
                            if agent_opinion.get('추천') == 'BUY':
                                buy_count += 1
                                confidence_sum += float(agent_opinion.get('신뢰도', 0))
                    symbol_scores.append((i, buy_count, confidence_sum))
                # BUY 개수, 신뢰도 합 기준 내림차순 정렬
                symbol_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
                top5 = symbol_scores[:5]
                top5_indices = [x[0] for x in top5]
                top5_symbols = [target_symbols_filtered[i] for i in top5_indices]
                top5_news = [news_analyses[i] for i in top5_indices]
                top5_strategies = [strategy_results[i] for i in top5_indices]
                top5_risks = [risk_analyses[i] for i in top5_indices]
                top5_techs = [technical_analyses[i] for i in top5_indices]
                now_ts = datetime.now().isoformat()
                top5_debates = []
                for idx, symbol in enumerate(top5_symbols):
                    news = top5_news[idx]
                    market_trend = 'neutral'
                    if isinstance(news, dict):
                        impact = news.get('시장영향도', 0.0)
                        if impact > 0:
                            market_trend = 'bullish'
                        elif impact < 0:
                            market_trend = 'bearish'
                    market_condition_obj = MarketCondition(
                        market_trend=market_trend,
                        volatility_level='high' if news.get('뉴스갯수', 0) >= 10 else 'low',
                        trading_volume=0.0,
                        sector_performance={},
                        major_events=[],
                        timestamp=datetime.now()
                    )
                    debate_context = {
                        'symbol': symbol,
                        'market_conditions': market_condition_obj,
                        'period': '1y',
                        'timestamp': now_ts,
                        'technical_indicators': top5_techs[idx].get('indicators', {}) if idx < len(top5_techs) else {},
                        'market_data': top5_techs[idx].get('raw_data', None) if idx < len(top5_techs) else None
                    }
                    debate_opinions = self.agent_debate_round(debate_context)
                    top5_debates.append(debate_opinions)
                await self._generate_investment_report(
                    top5_symbols,
                    top5_news,
                    top5_strategies,
                    top5_risks,
                    top5_techs,
                    top5_debates
                )
                self.save_debate_log(top5_symbols, top5_debates)
                # holdings 갱신: top5 포트폴리오에서 현금 제외, 종목코드-수량 dict로 저장
                # (구매개수 계산은 _generate_investment_report와 동일하게 적용)
                # 기술적 분석에서 symbol, current_price 활용
                allocations, cash_ratio = self._calculate_portfolio_allocation(top5_symbols, top5_debates or [[] for _ in top5_symbols])
                total_invest = 1_000_000
                code_to_price = {}
                for tech in top5_techs:
                    code = tech.get('symbol')
                    price = tech.get('current_price', 0)
                    if code and price:
                        code_to_price[code] = price
                holdings_new = {}
                for code in top5_symbols:
                    ratio = allocations.get(code, 0)
                    price = code_to_price.get(code, 0)
                    invest_amt = int(total_invest * ratio)
                    if price > 0:
                        qty = invest_amt // int(price)
                    else:
                        qty = 0
                    if qty > 0:
                        holdings_new[code] = qty
                self._save_holdings(holdings_new)
                self.is_running = False
                return

                # 4. 상태 업데이트 및 대기
                self.last_analysis_time = datetime.now()
                await self._wait_for_next_analysis()

            except Exception as e:
                self.logger.error(f"주기적 분석 중 오류 발생: {str(e)}")
                self.logger.error("시스템을 중단합니다.")
                self.is_running = False
                raise

    async def _generate_investment_report(self, symbols, news_analyses, strategies, risk_analyses, technical_analyses, debate_results=None):
        """
        상위 10개 종목 전체 분석 결과를 표/리스트로 구조화해 프롬프트에 포함, 보고서 파일로 저장
        """
        import openai
        import os
        from datetime import datetime
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            self.logger.error("OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다.")
            return
        client = openai.OpenAI(api_key=api_key)

        def make_table(rows, columns, code_to_name=None):
            table = '| ' + ' | '.join(columns) + ' |\n'
            table += '| ' + ' | '.join(['---']*len(columns)) + ' |\n'
            for row in rows:
                # row가 dict가 아니면 dict로 변환
                if not isinstance(row, dict):
                    if hasattr(row, 'dict'):
                        row = row.dict()
                    elif hasattr(row, 'to_dict'):
                        row = row.to_dict()
                    else:
                        row = {col: getattr(row, col, '') for col in columns}
                # 종목 코드 → 종목명 변환
                if code_to_name and '종목' in row:
                    code = row['종목']
                    name = code_to_name.get(code, code)
                    if not name or name in ["", None]:
                        name = "알 수 없음"
                    row['종목'] = f"{name} ({code})"
                table += '| ' + ' | '.join(str(row.get(col, '')) for col in columns) + ' |\n'
            return table

        # 종목코드→종목명 매핑 생성
        code_to_name = {code: self._get_stock_name(code) for code in symbols}

        # 각 분석 결과를 표/리스트로 요약 (상위 5개만)
        news_rows = news_analyses[:5]
        strategy_rows = strategies[:5]
        risk_rows = risk_analyses[:5]
        tech_rows = technical_analyses[:5]
        news_table = make_table(news_rows, ['종목', '감성점수', '시장영향도', '키워드'], code_to_name)
        strategy_table = make_table(strategy_rows, ['종목', '추천전략', '설명'], code_to_name)
        risk_table = make_table(risk_rows, ['종목', '리스크점수', '변동성', '리스크레벨'], code_to_name)
        tech_table = make_table(tech_rows, ['종목', 'RSI', 'MACD', '볼린저밴드상단', '볼린저밴드하단', '최근종가'], code_to_name)

        # 종목별 요약 설명 생성
        def make_summary_rows(symbols, news_analyses):
            summary_lines = []
            for code, news in zip(symbols, news_analyses):
                name = code_to_name.get(code, code)
                keywords = news.get('키워드', '') if isinstance(news, dict) else ''
                impact = news.get('시장영향도', '') if isinstance(news, dict) else ''
                summary_lines.append(f"- **{name} ({code})**: 주요 키워드: {keywords}, 시장영향도: {impact}")
            return '\n'.join(summary_lines)
        news_summary = make_summary_rows(symbols[:5], news_analyses[:5])

        # 리스크 분석 결과 디버깅용 로그
        for symbol, risk in zip(symbols, risk_analyses):
            self.logger.info(f"[DEBUG] {symbol} 리스크 분석 결과: {risk}")

        # debate 결과 기반 포트폴리오 비중 산출 함수
        allocations, cash_ratio = self._calculate_portfolio_allocation(symbols, debate_results or [[] for _ in symbols])

        # 전체 투자금(원)
        total_invest = 1_000_000
        # 종목별 현재가 추출 (기술적 분석 결과에서)
        code_to_price = {}
        for tech in technical_analyses:
            code = tech.get('symbol')
            price = tech.get('current_price', 0)
            if code and price:
                code_to_price[code] = price
        # 종목코드→종목명 매핑 생성
        code_to_name = {code: self._get_stock_name(code) for code in symbols}
        # 구매 개수 및 투자금 계산
        portfolio_rows = []
        remain_cash = total_invest
        for code in symbols:
            name = code_to_name.get(code, code)
            ratio = allocations.get(code, 0)
            price = code_to_price.get(code, 0)
            invest_amt = int(total_invest * ratio)
            if price > 0:
                qty = invest_amt // int(price)
                used = qty * int(price)
            else:
                qty = 0
                used = 0
            remain_cash -= used
            portfolio_rows.append({
                '종목명': name,
                '구매개수': qty,
                '투자금': f"{used:,}",
                '현재가': f"{price:,}"
            })
        # 현금
        portfolio_rows.append({'종목명': '현금', '구매개수': '-', '투자금': f"{remain_cash:,}", '현재가': '-'})
        portfolio_table = make_table(portfolio_rows, ['종목명', '구매개수', '투자금', '현재가'])

        # 프롬프트 템플릿(6번 항목에 LLM이 직접 논리적 이유와 기대효과를 작성하도록 지시)
        prompt = f"""
        국내 주식 자동매매 시스템에서 최근 거래량이 가장 많은 상위 10개 종목을 대상으로, 아래 항목별로 포트폴리오 투자 분석 보고서를 작성하세요.

        - 각 항목은 **별도의 소제목**으로 구분하고, 실제 데이터(표, 수치, 차트 등)와 논리적 근거를 최대한 활용해 상세히 기술하세요.
        - 각 종목별 분석 결과는 표 또는 리스트로 정리하세요.

        1. **전략 분석**
        {strategy_table}
        2. **뉴스/이슈 분석**
        {news_table}
        {news_summary}
        3. **리스크 분석**
        {risk_table}
        4. **기술적 분석**
        {tech_table}
        5. **시계열(통계/머신러닝 기반) 분석**
        (각 종목별 시계열 예측 결과를 요약해 표로 정리)
        6. **포트폴리오 구성 이유 및 기대효과**
        위 표와 분석 결과를 바탕으로, 각 종목의 선정 이유와 전체 포트폴리오의 기대효과를 논리적이고 구체적으로 작성하세요. (예: 각 종목이 포트폴리오에 포함된 근거, 산업/이슈/리스크/성장성 등 다양한 관점에서 설명, 분산투자 및 리스크 관리 효과, 기대 수익 등)
        
        **최종 포트폴리오 구성**
        {portfolio_table}
        
        분석은 최근 데이터(최소 최근 1년 이내)를 기준으로 하며, 데이터가 부족한 부분은 추정 근거를 명확히 밝혀 주세요.
        """
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            report_md = response.choices[0].message.content
            report_dir = "investment_reports"
            os.makedirs(report_dir, exist_ok=True)
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(report_dir, f"investment_report_portfolio_{now}.md")
            with open(filename, "w", encoding="utf-8") as f:
                f.write(report_md)
            self.logger.info(f"포트폴리오 투자 분석 보고서가 {filename}에 저장되었습니다.")
        except Exception as e:
            self.logger.error(f"투자 분석 보고서 생성 중 오류 발생: {str(e)}")

    async def _wait_for_next_analysis(self):
        """
        다음 분석까지 대기합니다.
        """
        try:
            # 시간프레임에 따른 대기 시간 설정
            if self.current_timeframe == "1m":
                await asyncio.sleep(60)
            elif self.current_timeframe == "5m":
                await asyncio.sleep(300)
            elif self.current_timeframe == "15m":
                await asyncio.sleep(900)
            elif self.current_timeframe == "1h":
                await asyncio.sleep(3600)
            elif self.current_timeframe == "1d":
                await asyncio.sleep(86400)
            else:
                await asyncio.sleep(3600)  # 기본값 1시간
                
        except Exception as e:
            self.logger.error(f"대기 중 오류 발생: {str(e)}")
            raise 

    async def analyze_stock(self, symbol: str) -> Dict:
        """
        주식 분석을 수행합니다.
        """
        try:
            self.logger.info(f"{symbol} 분석 시작")
            
            # 1. 뉴스 분석
            self.logger.info(f"{symbol} 뉴스 분석 시작")
            news_analysis = await self.news_analyzer.analyze(symbol)
            self.logger.info(f"{symbol} 뉴스 분석 완료")
            
            # 2. 전략 수립
            self.logger.info(f"{symbol} 전략 수립 시작")
            strategy = await self.strategy_generator.generate_strategy(
                user_input="",  # 빈 문자열로 설정
                market_conditions=news_analysis,  # 뉴스 분석 결과에서 자동으로 생성
                risk_tolerance=0.5,  # 기본값
                time_horizon=TimeHorizon.MEDIUM_TERM  # 명시적으로 설정
            )
            self.logger.info(f"{symbol} 전략 수립 완료")
            
            # 3. 리스크 분석
            self.logger.info(f"{symbol} 리스크 분석 시작")
            risk_analysis = await self.risk_analyzer.analyze(symbol, strategy)
            self.logger.info(f"{symbol} 리스크 분석 완료")
            
            # 4. 기술적 분석 (전략에서 선정된 종목들에 대해)
            self.logger.info(f"{symbol} 기술적 분석 시작")
            technical_analyses = []
            if not strategy.target_assets:
                self.logger.warning("분석할 종목이 없습니다.")
            else:
                # 종목코드 기준으로만 분석, 중복 제거
                symbols_to_analyze = set()
                for symbol in strategy.target_assets:
                    symbol_code = self._get_stock_code(symbol)
                    if symbol_code:
                        symbols_to_analyze.add(symbol_code)
                    else:
                        self.logger.warning(f"{symbol}의 종목코드 변환에 실패했습니다.")
                for symbol_code in symbols_to_analyze:
                    # 뉴스 분석 결과에 포함된 종목만 분석
                    symbol_name = self._get_stock_name(symbol_code)
                    if symbol_name not in news_symbols:
                        self.logger.info(f"{symbol_code}({symbol_name})는 뉴스 분석 결과에 없으므로 기술적 분석을 건너뜁니다.")
                        continue
                    try:
                        symbol_yf = symbol_code + ".KS"
                        analysis = self.technical_analyzer.analyze(symbol_yf)
                        if analysis:
                            technical_analyses.append(analysis)
                        else:
                            self.logger.warning(f"{symbol_code}의 기술적 분석에 실패했습니다.")
                    except Exception as e:
                        self.logger.error(f"{symbol_code} 분석 중 오류 발생: {str(e)}")
                        continue
            if not technical_analyses:
                self.logger.warning("기술적 분석 결과가 없습니다.")
            else:
                self.logger.info("기술적 분석 완료")

            # 5. 리스크 분석 (전략 결과를 인풋으로)
            # 기술적 분석 결과에서 market_data, technical_indicators 추출
            if technical_analyses and isinstance(technical_analyses[0], dict):
                market_data = technical_analyses[0].get('raw_data', pd.DataFrame())
                technical_indicators = technical_analyses[0].get('indicators', {})
            else:
                market_data = pd.DataFrame()
                technical_indicators = {}
            risk_analysis = self.risk_analyzer.assess_risk(
                request=RiskAssessmentRequest(
                    market_data=market_data,
                    technical_indicators=technical_indicators,
                    risk_tolerance=0.5,
                    portfolio_value=1000000.0,
                    positions=[],
                    market_context=None
                )
            )
            if not risk_analysis:
                raise ValueError("리스크 분석에 실패했습니다.")
            self.logger.info("리스크 분석 완료")

            # 6. 토론 기반 최종 결정 의견 도출
            # 기술적 분석을 실제로 수행한 종목코드 리스트에서 첫 번째 종목코드를 사용
            analyzed_symbols = [symbol_code for symbol_code in symbols_to_analyze if self._get_stock_name(symbol_code) in news_symbols]
            if analyzed_symbols:
                debate_symbol = analyzed_symbols[0]
            else:
                debate_symbol = target_symbol if target_symbol and target_symbol.isdigit() else self._get_stock_code(target_symbol)

            debate_context = {
                'symbol': debate_symbol,
                'market_conditions': market_condition_obj  # news_json 전체가 아니라 변환된 객체
            }
            debate_opinions = self.agent_debate_round(debate_context)
            self.logger.info(f"토론 라운드 결과: {debate_opinions}")

            # 7. 최종 결정 (토론 결과와 각 분석 결과를 종합)
            decision = await self._make_final_decision(
                news_analysis,
                strategy,
                risk_analysis,
                technical_analyses
            )
            if not decision:
                raise ValueError("최종 결정 생성에 실패했습니다.")

            # 7-1. 투자 분석 보고서 자동 생성
            await self._generate_investment_report(symbol, news_analysis, strategy, risk_analysis, technical_analyses)

            # 8. 결정 실행
            await self._execute_decision(decision)

            # 9. 상태 업데이트
            self.last_decision = decision
            self.last_analysis_time = datetime.now()

            # 10. 대기
            await self._wait_for_next_analysis()

            # 11. 최종 분석 결과 종합
            self.logger.info(f"{symbol} 최종 분석 결과 종합 시작")
            analysis_result = {
                'symbol': symbol,
                'news_analysis': news_analysis,
                'strategy': strategy,
                'risk_analysis': risk_analysis,
                'technical_analysis': technical_analyses[0] if technical_analyses else None,
                'market_context': market_context,
                'timestamp': datetime.now().isoformat()
            }
            self.logger.info(f"{symbol} 최종 분석 결과 종합 완료")
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"{symbol} 분석 중 오류 발생: {str(e)}")
            return {}

    def _get_stock_code(self, symbol: str) -> str:
        """
        종목명을 종목코드로 변환합니다. (FinanceDataReader 기반 자동 매핑)
        """
        try:
            # 종목코드가 이미 숫자로만 이루어진 경우
            if symbol.isdigit():
                return symbol
            # 자동 매핑
            code = self.name_to_code.get(symbol)
            if code:
                return code
            # 부분 일치(공백 제거, 대소문자 무시)
            for name, code in self.name_to_code.items():
                if name.replace(' ', '').lower() == symbol.replace(' ', '').lower():
                    return code
            return None
        except Exception as e:
            self.logger.error(f"종목코드 변환 중 오류 발생: {str(e)}")
            return None

    def _get_stock_name(self, code: str) -> str:
        """
        종목코드를 종목명으로 변환합니다. (FinanceDataReader 기반)
        """
        try:
            if self.stock_df is not None:
                result = self.stock_df[self.stock_df['Code'] == code]
                if not result.empty:
                    return result.iloc[0]['Name']
            return code
        except Exception as e:
            self.logger.error(f'종목명 변환 중 오류 발생: {str(e)}')
            return code

    def agent_debate_round(self, context):
        """
        에이전트 토론 라운드: 각 에이전트의 debate 결과만 수집해 반환합니다.
        """
        agents = [
            self.technical_analyzer,
            self.news_analyzer,
            self.strategy_generator,
            self.risk_analyzer
        ]
        final_opinions = []
        for agent in agents:
            if hasattr(agent, 'debate'):
                op = agent.debate(context, [])
                final_opinions.append(op)
            else:
                op = {'agent': agent.__class__.__name__, '추천': 'HOLD', '신뢰도': 0.5, '주장': '기본값', '핵심지표': '-', '전문가설명': '-'}
                final_opinions.append(op)
        self.logger.info(f"[디베이트] 최종 토론 결과: {final_opinions}")
        return final_opinions

    async def _make_final_decision(
        self,
        news_analysis: Dict,
        strategy: Dict,
        risk_analysis: Dict,
        technical_analyses: List[Dict]
    ) -> DecisionResponse:
        """
        최종 투자 결정을 생성합니다.
        """
        try:
            if not all([news_analysis, strategy, risk_analysis]):
                raise ValueError("필수 분석 데이터가 누락되었습니다.")
            
            # 리스크 파라미터 설정
            risk_parameters = self._create_risk_parameters(risk_analysis)
            
            # 모든 에이전트의 분석 결과를 종합
            agent_analyses = []
            
            # 뉴스 분석 결과 추가
            if isinstance(news_analysis, dict):
                agent_analyses.append(
                    AgentAnalysis(
                        agent_name="news_analyzer",
                        analysis_type="news",
                        confidence_score=news_analysis.get('confidence', 0.0),
                        recommendation=news_analysis,
                        timestamp=datetime.now()
                    )
                )
            
            # 전략 분석 결과 추가
            if isinstance(strategy, dict):
                agent_analyses.append(
                    AgentAnalysis(
                        agent_name="strategy_generator",
                        analysis_type="strategy",
                        confidence_score=strategy.get('confidence', 0.0),
                        recommendation=strategy,
                        timestamp=datetime.now()
                    )
                )
            
            # 리스크 분석 결과 추가
            if isinstance(risk_analysis, dict):
                agent_analyses.append(
                    AgentAnalysis(
                        agent_name="risk_analyzer",
                        analysis_type="risk",
                        confidence_score=risk_analysis.get('confidence', 0.0),
                        recommendation=risk_analysis,
                        timestamp=datetime.now()
                    )
                )
            
            # 기술적 분석 결과 추가
            if technical_analyses:
                for analysis in technical_analyses:
                    if isinstance(analysis, dict):
                        agent_analyses.append(
                            AgentAnalysis(
                                agent_name="technical_analyzer",
                                analysis_type="technical",
                                confidence_score=analysis.get('confidence', 0.0),
                                recommendation=analysis,
                                timestamp=datetime.now()
                            )
                        )
            
            # 시장 상황 분석
            market_context = await self._analyze_market_context(
                news_analysis,
                strategy,
                risk_analysis,
                technical_analyses
            )
            
            # 현재 가격 설정
            current_price = 0.0
            if technical_analyses and isinstance(technical_analyses[0], dict):
                current_price = technical_analyses[0].get('current_price', 0.0)
            
            # 최종 결정 요청
            request = DecisionRequest(
                symbol=self.current_symbol,
                timeframe=self.current_timeframe,
                current_price=current_price,
                market_context=market_context,
                risk_parameters=risk_parameters,
                agent_analyses=agent_analyses,
                technical_analysis=technical_analyses[0] if technical_analyses else None,
                news_analysis=news_analysis if isinstance(news_analysis, dict) else news_analysis.dict() if hasattr(news_analysis, 'dict') else {},
                strategy=strategy.dict() if hasattr(strategy, 'dict') else strategy,
                risk_assessment=risk_analysis.dict() if hasattr(risk_analysis, 'dict') else risk_analysis if isinstance(risk_analysis, dict) else {}
            )
            
            decision = self.decision_maker.make_decision(request)
            if not decision:
                raise ValueError("결정 생성에 실패했습니다.")
            
            return decision
            
        except Exception as e:
            self.logger.error(f"최종 결정 생성 중 오류 발생: {str(e)}")
            raise

    async def _analyze_market_context(
        self,
        news_analysis: Dict,
        strategy: Dict,
        risk_analysis: Dict,
        technical_analyses: List[Dict]
    ) -> MarketContext:
        """
        시장 상황을 분석합니다.
        """
        try:
            # 뉴스 감성과 시장 상황 통합
            market_sentiment = news_analysis.get('market_sentiment', 'NEUTRAL') if isinstance(news_analysis, dict) else 'NEUTRAL'
            
            # 기술적 분석에서 변동성과 추세 강도 추출
            volatility_level = 0.0
            trend_strength = 0.0
            if technical_analyses and isinstance(technical_analyses[0], dict):
                volatility_level = technical_analyses[0].get('volatility', 0.0)
                trend_strength = technical_analyses[0].get('trend_strength', 0.0)
            
            # 거래량 프로필 생성
            volume_profile = {}
            if technical_analyses and isinstance(technical_analyses[0], dict):
                volume_profile = technical_analyses[0].get('volume_profile', {})
            
            # 전략에서 시장 상황 추출
            market_condition = strategy.get('market_condition', 'UNKNOWN') if isinstance(strategy, dict) else 'UNKNOWN'
            
            return MarketContext(
                market_condition=market_condition,
                volatility_level=volatility_level,
                trend_strength=trend_strength,
                volume_profile=volume_profile,
                market_sentiment=market_sentiment
            )
            
        except Exception as e:
            self.logger.error(f"시장 상황 분석 중 오류 발생: {str(e)}")
            return MarketContext(
                market_condition="UNKNOWN",
                volatility_level=0.0,
                trend_strength=0.0,
                volume_profile={},
                market_sentiment="NEUTRAL"
            )

    async def _execute_decision(self, decision: DecisionResponse):
        """
        결정을 실행합니다.
        """
        try:
            if not decision:
                raise ValueError("실행할 결정이 없습니다.")
                
            if decision.error_message:
                self.logger.error(f"결정 실행 실패: {decision.error_message}")
                return
            
            self.logger.info("=== 결정 실행 시작 ===")
            self.logger.info(f"결정 타입: {decision.decision}")
            self.logger.info(f"신뢰도: {float(decision.confidence):.2f}")
            self.logger.info(f"포지션 크기: {float(decision.position_size):.1f}%")
            self.logger.info(f"진입가: {float(decision.entry_price):,.0f}원")
            self.logger.info(f"목표가: {float(decision.target_price):,.0f}원")
            self.logger.info(f"손절가: {float(decision.stop_loss):,.0f}원")
            self.logger.info(f"예상 수익률: {float(decision.expected_return):.1f}%")
            self.logger.info(f"최대 손실률: {float(decision.max_loss):.1f}%")
            
            self.logger.info("=== 결정 근거 ===")
            for reason in decision.reasons:
                self.logger.info(f"- {reason}")
            
            # 결정 실행 로직
            if decision.decision == DecisionType.BUY:
                self.logger.info("매수 신호 실행")
                # 매수 로직 구현
            elif decision.decision == DecisionType.SELL:
                self.logger.info("매도 신호 실행")
                # 매도 로직 구현
            elif decision.decision == DecisionType.HOLD:
                self.logger.info("홀드 신호 실행")
                # 홀드 로직 구현
            else:
                self.logger.info("대기 신호 실행")
                # 대기 로직 구현
            
            self.logger.info("=== 결정 실행 완료 ===")
            
        except Exception as e:
            self.logger.error(f"결정 실행 중 오류 발생: {str(e)}")
            raise

    # debate 결과를 텍스트 파일로 구조화해 저장
    def save_debate_log(self, symbols, debate_results):
        """debate 결과를 debate_logs/debate_log_YYYYMMDD.json에 저장"""
        from datetime import datetime
        out_dir = 'debate_logs'
        os.makedirs(out_dir, exist_ok=True)
        date_str = datetime.now().strftime('%Y%m%d')
        out_path = os.path.join(out_dir, f'debate_log_{date_str}.json')
        data = {s: d for s, d in zip(symbols, debate_results)}
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self.logger.info(f"debate 로그가 {out_path}에 저장되었습니다.")

    def print_debate_results(self, debate_results):
        """
        debate 결과(List[Dict])를 사람이 읽기 쉬운 텍스트 표로 구조화하여 출력합니다.
        """
        if not debate_results or not isinstance(debate_results, list):
            print("[디베이트 결과 없음]")
            return
        print("\n[에이전트 디베이트 결과 요약]")
        print("="*60)
        for res in debate_results:
            분야 = res.get("분야", res.get("agent", "-"))
            print(f"[분야] {분야}")
            print(f"  - 핵심지표: {res.get('핵심지표', '-')}")
            print(f"  - 주장: {res.get('주장', '-')}")
            print(f"  - 추천: {res.get('추천', '-')}  |  신뢰도: {res.get('신뢰도', '-')}\n")
            print(f"  - 전문가설명: {res.get('전문가설명', '-')}\n")
            print("-"*60)

    def _load_holdings(self, path='holdings.json'):
        """보유 종목/수량을 holdings.json에서 불러옴"""
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return {}
        except Exception as e:
            self.logger.error(f"보유 종목 불러오기 오류: {str(e)}")
            return {}

    def _save_holdings(self, holdings, path='holdings.json'):
        """보유 종목/수량을 holdings.json에 저장"""
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(holdings, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"보유 종목 저장 오류: {str(e)}")

    def save_forecast_csv(self, technical_analyses, date_str=None):
        """기술적 분석 시계열 예측 결과를 forecasts/forecast_YYYYMMDD.csv로 저장"""
        if date_str is None:
            from datetime import datetime
            date_str = datetime.now().strftime('%Y%m%d')
        rows = []
        for tech in technical_analyses:
            code = tech.get('symbol')
            name = self._get_stock_name(code)
            forecast = tech.get('price_forecast', {})
            dates = forecast.get('forecast_dates', [])
            prices = forecast.get('forecast_prices', [])
            for d, p in zip(dates, prices):
                rows.append({'종목코드': code, '종목명': name, '예측일': d, '예측가격': p})
        if rows:
            out_dir = 'forecasts'
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f'forecast_{date_str}.csv')
            with open(out_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['종목코드', '종목명', '예측일', '예측가격'])
                writer.writeheader()
                writer.writerows(rows)
            self.logger.info(f"시계열 예측 결과가 {out_path}에 저장되었습니다.")

    def _calculate_portfolio_allocation(self, symbols, debate_results):
        """
        debate_results: List[List[Dict]] (각 종목별 에이전트 의견 리스트)
        """
        allocations = {}
        total_score = 0.0
        for symbol, debates in zip(symbols, debate_results or [[]]*len(symbols)):
            buy_score = sum(1.0 * (d.get('decision') == 'BUY') * d.get('confidence', 0.5) for d in debates)
            hold_score = sum(0.5 * (d.get('decision') == 'HOLD') * d.get('confidence', 0.5) for d in debates)
            sell_score = sum(-1.0 * (d.get('decision') == 'SELL') * d.get('confidence', 0.5) for d in debates)
            score = max(buy_score + hold_score + sell_score, 0.0)
            allocations[symbol] = score
            total_score += score
        avg_score = total_score / (len(symbols) or 1)
        cash_ratio = max(0.1, 1.0 - total_score) if total_score < 1.0 else 0.05
        if total_score > 0:
            for symbol in allocations:
                allocations[symbol] = round((allocations[symbol] / total_score) * (1-cash_ratio), 3)
        else:
            for symbol in allocations:
                allocations[symbol] = round(0.9 / len(symbols), 3)
        return allocations, round(cash_ratio, 3)

if __name__ == "__main__":
    import asyncio
    manager = SystemManager()
    # 예시: 삼성전자, 1시간봉
    asyncio.run(manager.start_system(symbol="005930", timeframe="1h"))
    