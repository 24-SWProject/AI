from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType

# Milvus 서버에 연결
connections.connect(host='localhost', port='19530')

# 컬렉션 이름
collection_name = "sw_project_ai"

# 컬렉션 객체 생성
collection = Collection(collection_name)

# 컬렉션 삭제
collection.drop()
print(f"컬렉션 '{collection_name}'이 삭제되었습니다.")

# 새로운 필드 스키마 정의
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=3000),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=9000),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024)
]

schema = CollectionSchema(fields=fields, description="어디가유_데이트 코스 추천 ai")

# 컬렉션 재생성
collection = Collection(name=collection_name, schema=schema)
print(f"컬렉션 '{collection_name}'이 새로 생성되었습니다.")
