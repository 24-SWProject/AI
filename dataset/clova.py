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
            f"카테고리는 축제입니다. "
            f"축제 이름은 '{data.get('title', '정보없음')}', "
            f"축제 장소는 {data.get('place', '정보없음')}, "
            f"축제 시작 일자는 {data.get('openDate', '정보없음')}, "
            f"축제 종료 일자는 {data.get('endDate', '정보없음')}, "
            f"축제 대상은 {data.get('useTrgt', '정보없음')}입니다. "
        )
        return text_for_embedding
    
    
    @staticmethod
    def create_chunked_performance(data):
        text_for_embedding = (
            f"카테고리는 공연, 장르는 {data.get('category', '정보없음')}입니다. "
            f"공연의 제목은 '{data.get('title', '정보없음')}', "
            f"공연 장소는 {data.get('place', '정보없음')}, "
            f"공연 시작 일자는 {data.get('openDate', '정보없음')}, "
            f"공연 종료 일자는 {data.get('endDate', '정보없음')}입니다. "
        )
        return text_for_embedding

    @staticmethod
    def create_chunked_food(data):
        text_for_embedding = (
            f"카테고리는 음식점, 종류는 {data.get('majorCategory', '정보없음')}입니다. "
            f"음식점의 이름은 '{data.get('title', '정보없음')}', "
            f"음식점의 전화번호는 {data.get('phoneNumber', '정보없음')}, "
            f"자치구는 {data.get('guName', '정보없음')}, "
            f"상세 주소는 {data.get('address', '정보없음')}에 위치해있습니다."
        )
        return text_for_embedding

    @staticmethod
    def create_chunked_movie(data):
        text_for_embedding = (
            f"카테고리는 행사, "
            f"영화 제목은 {data.get('movieNm', '정보없음')}, "
            f"박스오피스 순위는 {data.get('rank', '정보없음')}, "
            f"개봉 일자는 {data.get('openDt', '정보없음')}, "
            f"누적 관객수는 {data.get('audiAcc', '정보없음')}명입니다."
            f"키워드는 영화입니다."
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