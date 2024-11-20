import pymysql
from pymilvus import connections, utility
import os
from dotenv import load_dotenv

load_dotenv()

# MySQL 연결 관리
class MySQLConnectionManager:
    def __init__(self):
        self.connection = None

    def connect(self):
        if self.connection is None:
            self.connection = pymysql.connect(
                host=os.environ.get('DATABASE_HOST'),
                user=os.environ.get('DATABASE_USERNAME'),
                password=os.environ.get('DATABASE_PASSWORD'),
                database=os.environ.get('DATABASE_NAME'),
                port=int(os.environ.get('DATABASE_PORT')),
                cursorclass=pymysql.cursors.DictCursor
            )

    def execute_query(self, query, params=None):
        try:
            self.connect()
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                return cursor.fetchall()
        except pymysql.MySQLError as e:
            print(f"MySQL 쿼리 실행 중 오류 발생: {e}")
            raise
        finally:
            self.close()

    def close(self):
        if self.connection:
            self.connection.close()
            self.connection = None


# Milvus 연결 관리
class MilvusConnectionManager:
    def connect(self):
        try:
            connections.connect(
                alias=os.environ.get('MILVUS_ALIAS'),
                host=os.environ.get('MILVUS_HOST'),
                port=int(os.environ.get('MILVUS_PORT'))
            )
            print("Milvus에 성공적으로 연결되었습니다.")
        except Exception as e:
            print(f"Milvus 연결 오류: {e}")
            raise

    def disconnect(self):
        try:
            connections.disconnect(alias=os.environ.get('MILVUS_ALIAS'))
            print("Milvus 연결이 해제되었습니다.")
        except Exception as e:
            print(f"Milvus 연결 해제 오류: {e}")

    def has_collection(self, collection_name):
        self.connect()
        return utility.has_collection(collection_name)

    def drop_collection(self, collection_name):
        self.connect()
        if self.has_collection(collection_name):
            utility.drop_collection(collection_name)
            print(f"컬렉션 '{collection_name}'이 삭제되었습니다.")
        else:
            print(f"컬렉션 '{collection_name}'이 존재하지 않습니다.")
