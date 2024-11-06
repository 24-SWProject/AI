import time
import requests
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from tqdm import tqdm
from clova import *
from dotenv import load_dotenv
import os

load_dotenv()


# Milvus 연결
def connect_to_milvus():
    try:
        connections.connect(alias="default", host='3.35.133.197', port='19530')
        print("Milvus에 성공적으로 연결되었습니다.")
    except Exception as e:
        print(f"Milvus 연결 오류: {e}")

# 1. fetch(mySQL)

# 1. fetch(mySQL)
def fetch_festival_data():
    url = os.environ.get('FESTIVAL_URL')
    results = requests.get(url)
    if results.status_code == 200:
        return results.json()
    else:
        return {"error": "데이터를 가져오는 데 실패했습니다."}

# 2. chunking
def chunked_festival_data(embedding_executor):
    results = fetch_festival_data()
    return [embedding_executor.create_chunked_festival(item) for item in results]

# 3. Embedding
def embedding_festival_data():
    embedding_executor = EmbeddingExecutor(
        host=os.environ.get('CLOVASTUDIO_EMBEDDING_HOST'),
        api_key=os.environ.get('CLOVASTUDIO_EMBEDDING_API_KEY'),
        api_key_primary_val=os.environ.get('CLOVASTUDIO_EMBEDDING_APIGW_API_KEY'),
        request_id=os.environ.get('CLOVASTUDIO_EMBEDDING_REQUEST_ID')
    )
    
    chunked_text_list = chunked_festival_data(embedding_executor)
    chunked_html = []

    for chunked_document in tqdm(chunked_text_list):
        try:
            response_data = embedding_executor.execute({"text": chunked_document})
            chunked_html.append({
                'text': chunked_document,
                'embedding': response_data,
                'source': chunked_document
            })
            time.sleep(1)
        except Exception as e:
            print(f"Unexpected error: {e}")

    print(chunked_html)
    return chunked_html

# 4. indexing
def indexing_festival_data():
    connect_to_milvus()
    collection_name = "festival_hereforus"

    # 기존 컬렉션 삭제 후 재생성
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"기존 컬렉션 '{collection_name}'을 삭제했습니다.")

    # 필드 및 스키마 정의
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=3000),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=9000),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024)
    ]
    schema = CollectionSchema(fields, description="sw_project")
    
    # 컬렉션 생성
    collection = Collection(name=collection_name, schema=schema, using='default', shards_num=2)
    print(f"컬렉션 '{collection_name}'이 생성되었습니다.")

    # 데이터 준비
    chunked_html = embedding_festival_data()
    source_list = []
    text_list = []
    embedding_list = []

    # 데이터를 entities 리스트에 추가
    for item in chunked_html:
        source_list.append(item['source'])
        text_list.append(item['text'])
        embedding_list.append(item['embedding'])

    entities = [source_list, text_list, embedding_list]
    
    # 인덱스 생성
    try:
        insert_result = collection.insert(entities)
        print("데이터 Insertion이 완료된 ID:", insert_result.primary_keys)
    except Exception as e:
        print(f"데이터 Insertion 오류: {e}")

    index_params = {
        "metric_type": "IP",
        "index_type": "HNSW",
        "params": {"M": 8, "efConstruction": 200}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    utility.index_building_progress("festival_hereforus")
    
    # print([index.params for index in collection.indexes])
    print("인덱스 생성이 완료되었습니다.")
    
    # 컬렉션 로드
    collection.load()
    print(f"컬렉션 '{collection_name}'이 로드되었습니다.")
    return collection
