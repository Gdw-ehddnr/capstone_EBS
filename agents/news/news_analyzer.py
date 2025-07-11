import urllib.request
import urllib.parse
import json
import ssl
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from difflib import SequenceMatcher
from deep_translator import GoogleTranslator
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from agents.utils.logger import AgentLogger
from agents.utils.call_openai_api import call_openai_api

client_id = "jWYb81zpxwjjBOvjTlc1"
client_secret = "CH69vSJ6hu"

class MarketImpactAnalyzer:
    def __init__(self):
        self.sector_keywords = {
            '반도체': ['반도체', '파운드리', 'DDR', 'DRAM', 'NAND', '웨이퍼'],
            '2차전지': ['2차전지', '배터리', 'LFP', 'NCM', '양극재', '음극재', '분리막'],
            '자동차': ['자동차', '전기차', 'EV', '내연기관', '하이브리드'],
            'IT': ['소프트웨어', '플랫폼', '클라우드', 'AI', '인공지능'],
            '바이오': ['제약', '바이오', '신약', '임상', '백신'],
            '금융': ['은행', '증권', '보험', '카드', '핀테크']
        }
        
        self.market_indicators = {
            '금리': ['기준금리', '국고채', '회사채', '금리인상', '금리인하'],
            '환율': ['원달러', '원화가치', '환율', '달러인덱스'],
            '원자재': ['유가', '구리', '리튬', '니켈', '코발트'],
            '경제지표': ['GDP', '물가', '고용', '수출', '무역수지']
        }

    def analyze_market_impact(self, news_text: str, stock_name: str) -> Dict:
        """시장 영향 분석"""
        impact_analysis = {
            'impact_level': 0.0,  # -1.0 ~ 1.0
            'confidence_score': 0.0,  # 0.0 ~ 1.0
            'affected_sectors': [],
            'market_indicators': [],
            'related_stocks': [],
            'time_horizon': 'short_term'  # short_term, mid_term, long_term
        }
        
        # 섹터 영향 분석
        for sector, keywords in self.sector_keywords.items():
            for keyword in keywords:
                if keyword in news_text:
                    if sector not in impact_analysis['affected_sectors']:
                        impact_analysis['affected_sectors'].append(sector)

        # 시장 지표 영향 분석
        for indicator, keywords in self.market_indicators.items():
            for keyword in keywords:
                if keyword in news_text:
                    if indicator not in impact_analysis['market_indicators']:
                        impact_analysis['market_indicators'].append(indicator)

        # 영향도 계산
        impact_analysis['impact_level'] = self._calculate_impact_level(news_text)
        impact_analysis['confidence_score'] = self._calculate_confidence_score(news_text)
        impact_analysis['time_horizon'] = self._determine_time_horizon(news_text)
        impact_analysis['related_stocks'] = self._find_related_stocks(news_text, stock_name)

        return impact_analysis

    def _calculate_impact_level(self, text: str) -> float:
        """뉴스의 시장 영향도 계산"""
        impact_keywords = {
            'positive': ['급등', '상승', '호실적', '수주', '흑자전환', '매출증가'],
            'negative': ['급락', '하락', '적자전환', '매출감소', '리스크', '우려']
        }
        
        score = 0.0
        for keyword in impact_keywords['positive']:
            if keyword in text:
                score += 0.2
        for keyword in impact_keywords['negative']:
            if keyword in text:
                score -= 0.2
                
        return max(min(score, 1.0), -1.0)

    def _calculate_confidence_score(self, text: str) -> float:
        """분석 신뢰도 점수 계산"""
        confidence_keywords = ['전망', '예상', '추정', '확인', '발표', '공시']
        score = 0.0
        
        for keyword in confidence_keywords:
            if keyword in text:
                score += 0.2
                
        return min(score, 1.0)

    def _determine_time_horizon(self, text: str) -> str:
        """영향 시간 범위 결정"""
        short_term = ['당일', '오늘', '내일', '이번주', '단기']
        mid_term = ['이번달', '다음달', '분기', '중기']
        long_term = ['내년', '장기', '중장기', '미래']
        
        for term in long_term:
            if term in text:
                return 'long_term'
        for term in mid_term:
            if term in text:
                return 'mid_term'
        return 'short_term'

    def _find_related_stocks(self, text: str, main_stock: str) -> List[str]:
        """연관 종목 찾기"""
        related = [main_stock]
        return related

class NewsAnalyzer:
    def __init__(self):
        self.logger = AgentLogger("news_analyzer")
        self.market_impact_analyzer = MarketImpactAnalyzer()

    def get_top100_by_volume(self):
        url = "https://finance.naver.com/sise/sise_quant.naver"
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url, headers=headers)
        context = ssl._create_unverified_context()
        res = urllib.request.urlopen(req, context=context)
        soup = BeautifulSoup(res, "html.parser")

        stock_names = []
        for item in soup.select("table.type_2 tr"):
            td = item.select("td")
            if len(td) > 1:
                name = td[1].get_text(strip=True)
                if name:
                    stock_names.append(name)
            if len(stock_names) >= 10:
                break
        return stock_names

    def is_similar(self, new_text, existing_texts, threshold=0.8):
        for text in existing_texts:
            similarity = SequenceMatcher(None, new_text, text).ratio()
            if similarity > threshold:
                return True
        return False

    def analyze_sentiment_for_stock(self, stock_name):
        encText = urllib.parse.quote(stock_name)
        displayNum = 20
        url = f"https://openapi.naver.com/v1/search/news?query={encText}&display={displayNum}&sort=sim"

        context = ssl._create_unverified_context()
        request = urllib.request.Request(url)
        request.add_header("X-Naver-Client-Id", client_id)
        request.add_header("X-Naver-Client-Secret", client_secret)

        try:
            response = urllib.request.urlopen(request, context=context)
        except:
            print(f"[요청 실패] {stock_name}")
            return None

        rescode = response.getcode()
        if rescode != 200:
            return None

        response_body = response.read().decode('utf-8')
        news_data = json.loads(response_body)

        analyzer = SentimentIntensityAnalyzer()
        sentiment_scores = []
        unique_descriptions = []
        latest_title = ""
        market_impacts = []

        for item in news_data['items']:
            title = item['title'].replace('<b>', '').replace('</b>', '')
            description = item['description'].replace('<b>', '').replace('</b>', '')

            if self.is_similar(description, unique_descriptions):
                    continue
            unique_descriptions.append(description)
            latest_title = title

            text_kr = f"{title}. {description}"
            
            # 시장 영향 분석
            market_impact = self.market_impact_analyzer.analyze_market_impact(text_kr, stock_name)
            market_impacts.append(market_impact)

            try:
                translated = GoogleTranslator(source='ko', target='en').translate(text_kr)
            except Exception as e:
                print(f"[번역 실패] {e}")
                continue
                
            sentiment = analyzer.polarity_scores(translated)
            compound_score = sentiment['compound']
            sentiment_scores.append(compound_score)

        if sentiment_scores:
            avg_score = sum(sentiment_scores) / len(sentiment_scores)
            buy_probability = int(((avg_score + 1) / 2) * 100)
            
            # 시장 영향 종합
            avg_impact_level = sum(impact['impact_level'] for impact in market_impacts) / len(market_impacts)
            affected_sectors = list(set([sector for impact in market_impacts for sector in impact['affected_sectors']]))
            
            return {
                "종목명": stock_name,
                "감성점수": round(avg_score, 3),
                "매수확률": buy_probability,
                "뉴스갯수": len(sentiment_scores),
                "최신기사제목": latest_title,
                "추천": "매수 추천" if avg_score > 0 else "매도 추천" if avg_score < 0 else "중립",
                "시장영향도": round(avg_impact_level, 3),
                "영향섹터": affected_sectors
            }
        else:
            return {
                "종목명": stock_name,
                "감성점수": None,
                "매수확률": None,
                "뉴스갯수": 0,
                "최신기사제목": "",
                "추천": "분석불가",
                "시장영향도": 0.0,
                "영향섹터": []
            }

    def run_sentiment_analysis(self) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        뉴스 데이터를 분석하여 감성 점수와 관련 메트릭을 계산합니다.
        
        Returns:
            Tuple[pd.DataFrame, List[Dict]]: 분석 결과 DataFrame과 JSON 형식의 결과
        """
        try:
            self.logger.info("Starting sentiment analysis")
            
            # 거래대금 상위 종목 가져오기
            top_stocks = self.get_top100_by_volume()
            self.logger.info(f"Retrieved top {len(top_stocks)} stocks by trading volume")
            
            # 각 종목별 뉴스 분석 실행
            results = []
            for stock in top_stocks:
                self.logger.info(f"Analyzing news for {stock}")
                result = self.analyze_sentiment_for_stock(stock)
                if result:
                    results.append(result)
                time.sleep(1.5)  # API 호출 제한 준수
            
            # DataFrame 생성
            df_result = pd.DataFrame(results)
            
            # JSON 형식으로 변환
            json_result = []
            for _, row in df_result.iterrows():
                stock_data = {
                    "종목명": row["종목명"],
                    "감성점수": float(row["감성점수"]) if pd.notnull(row["감성점수"]) else 0.0,
                    "매수확률": float(row["매수확률"]) if pd.notnull(row["매수확률"]) else 0.0,
                    "뉴스갯수": int(row["뉴스갯수"]),
                    "시장영향도": float(row["시장영향도"]),
                    "영향섹터": row["영향섹터"] if isinstance(row["영향섹터"], list) else [],
                    "추천": row["추천"],
                    "technical_indicators": None  # Technical Agent에서 추가될 예정
                }
                json_result.append(stock_data)
            
            self.logger.info(f"Sentiment analysis completed for {len(json_result)} stocks")
            # 최신 결과를 news_output.json에 저장
            output_path = os.path.join(self.pipeline_dir, "news_output.json") if hasattr(self, 'pipeline_dir') else "data/pipeline/news_output.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump({"stocks": json_result}, f, ensure_ascii=False, indent=2)
            return df_result, json_result
                    
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {str(e)}")
            raise

    def propose(self, context):
        """
        자신의 뉴스 분석 결과를 의견으로 제시합니다.
        """
        symbol = context.get('symbol', '삼성전자')
        analysis = self.analyze_sentiment_for_stock(symbol)
        sentiment_score = analysis.get('감성점수', 0.0) if analysis else 0.0
        market_impact = analysis.get('시장영향도', 0.0) if analysis else 0.0
        news_count = analysis.get('뉴스갯수', 0) if analysis else 0
        decision = 'HOLD'
        confidence = 0.5
        reasons = []
        # 감성점수와 시장영향도, 뉴스갯수 등 종합 판단
        if sentiment_score is not None and market_impact is not None:
            if sentiment_score > 0.5 and market_impact > 0.2 and news_count >= 5:
                decision = 'BUY'
                confidence = min(1.0, (sentiment_score + market_impact) / 2 + 0.2)
                reasons.append(f'강한 긍정 뉴스({sentiment_score:.2f}), 시장영향도({market_impact:.2f}), 뉴스갯수 {news_count}건')
            elif sentiment_score < -0.5 and market_impact < -0.2 and news_count >= 5:
                decision = 'SELL'
                confidence = min(1.0, abs(sentiment_score + market_impact) / 2 + 0.2)
                reasons.append(f'강한 부정 뉴스({sentiment_score:.2f}), 시장영향도({market_impact:.2f}), 뉴스갯수 {news_count}건')
            elif abs(sentiment_score) > 0.3 and news_count >= 3:
                decision = 'HOLD'
                confidence = min(0.7, abs(sentiment_score) + 0.1)
                reasons.append(f'약한 뉴스 신호({sentiment_score:.2f}), 뉴스갯수 {news_count}건')
            else:
                decision = 'HOLD'
                confidence = 0.5
                reasons.append(f'분석 신호 약함({sentiment_score:.2f}, {market_impact:.2f}, 뉴스 {news_count}건)')
        reason = ', '.join(reasons) if reasons else '뉴스 분석 결과'
        return {
            'agent': 'news_analyzer',
            'decision': decision,
            'confidence': confidence,
            'reason': reason
        }

    def debate(self, context, others_opinions, my_opinion_1st_round=None):
        analysis = self.analyze_sentiment_for_stock(context['symbol'])
        sentiment = analysis.get('감성점수', 0.0)
        impact = analysis.get('시장영향도', 0.0)
        news_count = analysis.get('뉴스갯수', 0)
        핵심지표 = {"감성점수": sentiment, "시장영향도": impact, "뉴스갯수": news_count}
        주장 = f"뉴스 감성점수 {sentiment:.2f}, 시장영향도 {impact:.2f}, 뉴스 {news_count}건."
        if sentiment > 0.5 and impact > 0.2:
            추천 = "BUY"
            신뢰도 = 0.8
            주장 += " 긍정적 뉴스가 많으므로 매수 추천."
        elif sentiment < -0.5 and impact < -0.2:
            추천 = "SELL"
            신뢰도 = 0.8
            주장 += " 부정적 뉴스가 많으므로 매도 추천."
        else:
            추천 = "HOLD"
            신뢰도 = 0.5
            주장 += " 중립적."
        # OpenAI API로 전문가 설명 생성
        prompt = f"""너는 뉴스 분석 전문가야. 아래 수치를 바탕으로 투자자에게 논리적으로 설명해줘.\n감성점수: {sentiment:.2f}, 시장영향도: {impact:.2f}, 뉴스갯수: {news_count}\n이 수치가 의미하는 바와 투자 판단에 미치는 영향, 추천 의견을 전문가답게 3~4문장으로 써줘."""
        전문가설명 = call_openai_api(prompt)  # 실제 OpenAI API 호출 함수 필요
        return {
            "agent": "news_analyzer",
            "분야": "뉴스",
            "핵심지표": 핵심지표,
            "주장": 주장,
            "추천": 추천,
            "신뢰도": 신뢰도,
            "전문가설명": 전문가설명
        }

if __name__ == "__main__":
    analyzer = NewsAnalyzer()
    df_result, json_result = analyzer.run_sentiment_analysis()