import json
import http.client
import requests
from dotenv import load_dotenv
import os

load_dotenv()


# embedding API
class EmbeddingExecutor:
    def __init__(self, host, api_key, api_key_primary_val, request_id):
        self._host = host
        self._api_key = api_key
        self._api_key_primary_val = api_key_primary_val
        self._request_id = request_id

    def _send_request(self, completion_request):
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'X-NCP-CLOVASTUDIO-API-KEY': self._api_key,
            'X-NCP-APIGW-API-KEY': self._api_key_primary_val,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id
        }

        conn = http.client.HTTPSConnection(self._host)
        conn.request('POST', os.environ.get('CLOVASTUDIO_EMBEDDING_URL'), json.dumps(completion_request), headers)
        response = conn.getresponse()
        result = json.loads(response.read().decode(encoding='utf-8'))
        conn.close()
        return result

    def execute(self, completion_request):
        res = self._send_request(completion_request)
        if res['status']['code'] == '20000':
            return res['result']['embedding']
        else:
            print(res)
            return 'Error'

    @staticmethod
    def create_chunked_festival(data):
        text_for_embedding = (
            f"카테고리: 축제, "
            f"축제 이름: {data.get('TITLE')}, "
            f"축제 장소: {data.get('PLACE')}, "  
            f"자치구: {data.get('GUNAME')}, "
            f"시작 일자: {data.get('STRTDATE')}, "
            f"종료 일자: {data.get('END_DATE')}, "
            f"예약 사이트: {data.get('ORG_LINK')}, "
            f"위치: ({data.get('LAT')}, {data.get('LOT')})"
        )
        return text_for_embedding

    @staticmethod
    def create_chunked_food(data):
        text_for_embedding = (
            f"카테고리: 음식점, "
            f"이름: {data.get('title')}, "
            f"전화번호: {data.get('phoneNumber')}, "
            f"구: {data.get('guName')}, "
            f"주소: {data.get('address')}, "
            f"위치: ({data.get('GPSx')}, {data.get('GPSy')})"
            f"키워드: {data.get('majorCategory')}, {data.get('subCategory')}"
        )
        return text_for_embedding

    @staticmethod
    def create_chunked_movie(data):
        text_for_embedding = (
            f"카테고리: 영화,"
            f"영화 제목: {data.get('movieNm')}, "
            f"박스오피스 순위: {data.get('rank')}, "
            f"개봉 일자: '{data.get('openDt')}, "
            f"누적 관객수: {data.get('audiAcc')}"
        )
        return text_for_embedding
    
    @staticmethod
    def create_chunked_weather(data):
        text_for_embedding = (
            f"카테고리: 날씨,"
            f"날씨 상태: {data.get('main')}, "
            f"상세 정보: {data.get('description')}, "
            f"아이콘: {data.get('icon')}, "
            f"현재 기온: '{data.get('temperature')}, "
            f"체감 온도: {data.get('feelsLike')}, "
            f"최소 기온: {data.get('tempMin')}, "
            f"최고 기온: {data.get('tempMax')}, "
            f"기압: {data.get('pressure')}, "
            f"습도: {data.get('humidity')}"
        )
        return text_for_embedding
    
    @staticmethod
    def create_chunked_performance(data):
        text_for_embedding = (
            f"카테고리: 공연, {data.get('genrenm')}, "
            f"공연 제목: {data.get('prfnm')}, "
            f"공연 시작 일자: {data.get('prfpdfrom')}, "
            f"공연 종료 일자: {data.get('prfpdto')}, "
            f"공연 장소: '{data.get('fcltynm')}, "
            f"포스터 이미지: {data.get('poster')}"
        )
        return text_for_embedding


# ClovaX API
class CompletionExecutor:
    def __init__(self, host, api_key, api_key_primary_val, request_id):
        self._host = host
        self._api_key = api_key
        self._api_key_primary_val = api_key_primary_val
        self._request_id = request_id
 
    def execute(self, completion_request):
        headers = {
            "X-NCP-CLOVASTUDIO-API-KEY": self._api_key,
            "X-NCP-APIGW-API-KEY": self._api_key_primary_val,
            "X-NCP-CLOVASTUDIO-REQUEST-ID": self._request_id,
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "text/event-stream"
        }
 
        response = requests.post(
            self._host + os.environ.get('CLOVASTUDIO_MODEL_URL'),
            headers=headers,
            json=completion_request,
            stream=True
        )
 
        # 스트림에서 마지막 'data:' 라인을 찾기 위한 로직
        last_data_content = ""
 
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode("utf-8")
                if '"data":"[DONE]"' in decoded_line:
                    break
                if decoded_line.startswith("data:"):
                    last_data_content = json.loads(decoded_line[5:])["message"]["content"]
 
        return last_data_content
            
def query_embed(text: str):
    embedding_executor = EmbeddingExecutor(
        host=os.environ.get('CLOVASTUDIO_EMBEDDING_HOST'),
        api_key=os.environ.get('CLOVASTUDIO_EMBEDDING_API_KEY'),
        api_key_primary_val=os.environ.get('CLOVASTUDIO_EMBEDDING_APIGW_API_KEY'),
        request_id=os.environ.get('CLOVASTUDIO_EMBEDDING_REQUEST_ID'),
    )

    request_data = {"text": text}
    response_data = embedding_executor.execute(request_data)

    return response_data