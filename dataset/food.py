import time
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from tqdm import tqdm
import pymysql
from dataset.clova import *
from dotenv import load_dotenv
import os
from time import sleep

load_dotenv()

# Milvus 연결
def connect_to_milvus():
    try:
        connections.connect(
            alias=os.environ.get('MILVUS_ALIAS'),
            host=os.environ.get('MILVUS_AWS_HOST'),
            port=int(os.environ.get("MILVUS_PORT")),
            max_send_message_length=512 * 1024 * 1024,  # 512MB
            max_receive_message_length=512 * 1024 * 1024  # 512MB
        )
        print("Milvus에 성공적으로 연결되었습니다.")
    except Exception as e:
        print(f"Milvus 연결 오류: {e}")

# 1. fetch(mySQL)
def fetch_food_data(offset=0, limit=1000):
    connection = pymysql.connect(
        host=os.environ.get('DATABASE_HOST'),
        user=os.environ.get('DATABASE_USERNAME'),
        password=os.environ.get('DATABASE_PASSWORD'),
        database=os.environ.get('DATABASE_NAME'),
        port=int(os.environ.get('DATABASE_PORT')),
        cursorclass=pymysql.cursors.DictCursor
    )
    try:
        with connection.cursor() as cursor:
            # 청크 단위로 데이터를 가져오는 쿼리
            sql = "SELECT id, title, phoneNumber, guName, address, gpsX, gpsY, majorCategory, subCategory FROM food LIMIT %s OFFSET %s"
            cursor.execute(sql, (limit, offset))  # 쿼리 실행
            results = cursor.fetchall()  # 쿼리 결과 가져오기
        return results
    finally:
        connection.close()

# 2. chunking
def chunked_food_data(embedding_executor, total_limit=1000):
    offset = 0
    limit = 1000
    fetched = 0
    all_chunked_text = []  # 모든 청크를 저장할 리스트

    # 데이터가 없을 때까지 반복적으로 가져오기
    while fetched < total_limit:
        results = fetch_food_data(offset, limit)
        if not results:  # 더 이상 데이터가 없으면 중지
            break

        # 현재 청크의 데이터를 처리하고 all_chunked_text에 추가
        chunked_text_list = [embedding_executor.create_chunked_food(item) for item in results]
        all_chunked_text.extend(chunked_text_list)

        offset += limit  # 다음 청크로 이동

    # print(all_chunked_text)
    return all_chunked_text[:total_limit]  # 모든 청크가 포함된 리스트 반환


# 3. Embedding
def embedding_food_data(total_limit=100):
    embedding_executor = EmbeddingExecutor(
        host=os.environ.get('CLOVASTUDIO_EMBEDDING_HOST'),
        api_key=os.environ.get('CLOVASTUDIO_EMBEDDING_API_KEY'),
        api_key_primary_val=os.environ.get('CLOVASTUDIO_EMBEDDING_APIGW_API_KEY'),
        request_id=os.environ.get('CLOVASTUDIO_EMBEDDING_REQUEST_ID')
    )
    
    chunked_text_list = chunked_food_data(embedding_executor, total_limit)
    chunked_html = []

    for chunked_document in tqdm(chunked_text_list):
        try:
            response_data = embedding_executor.execute({"text": chunked_document})
            chunked_html.append({
                'text': chunked_document,
                'embedding': response_data
            })
            time.sleep(1)
        except Exception as e:
            print(f"Unexpected error: {e}")

    # print(chunked_html)
    return chunked_html

# 4. indexing
def indexing_food_data(total_limit=100):
    connect_to_milvus()
    collection_name = "food_hereforus"

    # 기존 컬렉션 삭제 후 재생성
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"기존 컬렉션 '{collection_name}'을 삭제했습니다.")

    # 필드 및 스키마 정의
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=9000),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024)
    ]
    schema = CollectionSchema(fields, description="sw_project")
    
    # 컬렉션 생성
    collection = Collection(name=collection_name, schema=schema, using='default', shards_num=2)
    print(f"컬렉션 '{collection_name}'이 생성되었습니다.")

    # 데이터 준비
    chunked_html = embedding_food_data(total_limit)
    text_list = []
    embedding_list = []

    # 데이터를 entities 리스트에 추가
    for item in chunked_html:
        text_list.append(item['text'])
        embedding_list.append(item['embedding'])
        print(item)

    # Milvus에서 요구하는 형태로 데이터를 통합
    entities = [text_list, embedding_list]
    # 데이터 삽입
    
    try:
        insert_result = collection.insert(entities)
        print("데이터 Insertion이 완료된 ID:", insert_result.primary_keys)
    except Exception as e:
        print(f"데이터 Insertion 오류: {e}")

    # 인덱스 생성
    index_params = {
        "metric_type": "IP",
        "index_type": "HNSW",
        "params": {"M": 8, "efConstruction": 200}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    
    # 인덱스 생성 완료될 때까지 대기
    while True:
        progress = utility.index_building_progress(collection_name)
        if progress["state"] == "Finished":
            print("인덱스 생성이 완료되었습니다.")
            break
        else:
            print("인덱스 생성 중... 잠시 기다려 주세요.")
            sleep(1)
            
    # 컬렉션 로드
    collection.load()
    print(f"컬렉션 '{collection_name}'이 로드되었습니다.")
    return collection
