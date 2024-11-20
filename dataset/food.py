import time
import requests
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from tqdm import tqdm
from dataset.clova import *
from dotenv import load_dotenv
import os

load_dotenv(override=True)

# Milvus 연결
def connect_to_milvus():
    try:
        connections.connect(alias=os.environ.get('MILVUS_ALIAS'),
                            host=os.environ.get('MILVUS_HOST'),
                            port=int(os.environ.get("MILVUS_PORT")))
        print("Milvus에 성공적으로 연결되었습니다.")
    except Exception as e:
        print(f"Milvus 연결 오류: {e}")

# Milvus 컬렉션 설정
def setup_collection():
    collection_name = "food_hereforus"

    # 기존 컬렉션 삭제 후 재생성
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"기존 컬렉션 '{collection_name}'을 삭제했습니다.")

    # 필드 및 스키마 정의
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=9000),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024)
    ]
    schema = CollectionSchema(fields, description="sw_project")
    
    # 컬렉션 생성
    collection = Collection(name=collection_name, schema=schema, using='default', shards_num=2)
    print(f"컬렉션 '{collection_name}'이 생성되었습니다.")
    return collection

# 페이징 데이터 가져오기
def fetch_food_data(page, size, max_retries=5):
    url = f"{os.environ.get('FOOD_URL')}?page={page}&size={size}"
    retries = 0
    while retries < max_retries:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:  # Too many requests
            retry_after = int(response.headers.get("Retry-After", 5))  # 헤더 값 또는 기본값
            print(f"Rate limit exceeded. Retrying in {retry_after} seconds... (Attempt {retries + 1}/{max_retries})")
            time.sleep(retry_after)
            retries += 1
        else:
            print(f"HTTP {response.status_code}: {response.text}")
            break
    return {"error": "데이터를 가져오는 데 실패했습니다."}

# 데이터를 청크로 나누기
def process_batch(collection, page, size, embedding_executor):
    # 데이터 가져오기
    data = fetch_food_data(page, size)
    if "error" in data or not data.get("content"):
        print("No more data to process or an error occurred.")
        return False

    print(f"Processing page {page + 1}, size: {size}")
    chunked_data = []

    # 청크 생성
    for item in data["content"]:
        chunked_text = embedding_executor.create_chunked_food(item)
        chunked_data.append({
            "id": item["id"],
            "text": chunked_text
        })

    # Embedding 처리
    embedded_data = []
    for chunk in tqdm(chunked_data):
        try:
            embedding = embedding_executor.execute({"text": chunk["text"]})
            embedded_data.append({
                "id": chunk["id"],
                "text": chunk["text"],
                "embedding": embedding
            })
            time.sleep(1)
        except Exception as e:
            print(f"Embedding error for ID {chunk['id']}: {e}")

    # Milvus에 데이터 삽입
    ids = [item["id"] for item in embedded_data]
    texts = [item["text"] for item in embedded_data]
    embeddings = [item["embedding"] for item in embedded_data]

    try:
        collection.insert([ids, texts, embeddings])
        print(f"Batch {page + 1}: 데이터 삽입 완료")
    except Exception as e:
        print(f"Batch {page + 1}: 데이터 삽입 오류: {e}")

    # 다음 페이지 여부 확인
    return page + 1 < 1

# 메인 인덱싱 함수
def indexing_food_data_in_batches(batch_size=1000):
    connect_to_milvus()
    collection = setup_collection()

    embedding_executor = EmbeddingExecutor(
        host=os.environ.get('CLOVASTUDIO_EMBEDDING_HOST'),
        api_key=os.environ.get('CLOVASTUDIO_EMBEDDING_API_KEY'),
        api_key_primary_val=os.environ.get('CLOVASTUDIO_EMBEDDING_APIGW_API_KEY'),
        request_id=os.environ.get('CLOVASTUDIO_EMBEDDING_REQUEST_ID')
    )

    # 페이징 처리
    page = 0
    while True:
        more_data = process_batch(collection, page, batch_size, embedding_executor)
        if not more_data:
            print("모든 데이터 처리가 완료되었습니다.")
            break
        page += 1

    # 인덱스 생성
    index_params = {
        "metric_type": "IP",
        "index_type": "HNSW",
        "params": {"M": 8, "efConstruction": 200}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print("인덱스 생성 완료.")

    # 컬렉션 로드
    collection.load()
    print(f"컬렉션 '{collection.name}'이 로드되었습니다.")
