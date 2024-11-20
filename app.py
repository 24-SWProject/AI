from flask import Flask, request, jsonify
from pymilvus import Collection, utility
from dataset.food import *
from dataset.festival import * 
from dataset.clova import *
from dataset.performance import *
from dataset.movie import *
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

# Milvus 연결
def connect_to_milvus():
    try:
        connections.connect(alias=os.environ.get('MILVUS_ALIAS'),
                            host=os.environ.get('MILVUS_HOST'),
                            port=int(os.environ.get("MILVUS_PORT")))
        print("Milvus에 성공적으로 연결되었습니다.")
    except Exception as e:
        print(f"Milvus 연결 오류: {e}")

# festival data (cycle: every day)
@app.route('/festival', methods=['GET'])
def get_festival():
    try:
        indexing_festival_data()
        return jsonify({"message": "데이터가 성공적으로 임베딩되었습니다."}), 200
    except Exception as e: 
        return jsonify({"error": f"데이터를 가져오는 데 실패했습니다. 에러: {str(e)}"}), 400

# movie data (cycle: every day)
@app.route('/movie', methods=['GET'])
def get_weather():
    try:
        indexing_movie_data()
        return jsonify({"message": "데이터가 성공적으로 임베딩되었습니다."}), 200
    except Exception as e: 
        return jsonify({"error": f"데이터를 가져오는 데 실패했습니다. 에러: {str(e)}"}), 400
    
# food data (cycle: ?)
@app.route('/food', methods=['GET'])
def get_food():
    try:
        indexing_food_data_in_batches()
        return jsonify({"message": "데이터가 성공적으로 임베딩되었습니다."}), 200
    except Exception as e: 
        return jsonify({"error": f"데이터를 가져오는 데 실패했습니다. 에러: {str(e)}"}), 400
    
# performance data (cycle: every day)
@app.route('/performance', methods=['GET'])
def get_performance():
    try:
        indexing_performance_data()
        return jsonify({"message": "데이터가 성공적으로 임베딩되었습니다."}), 200
    except Exception as e: 
        return jsonify({"error": f"데이터를 가져오는 데 실패했습니다. 에러: {str(e)}"}), 400


def recommendByDB():
    # milvus 연결
    connect_to_milvus()

    # 요청 데이터 읽기
    data = request.get_json()
    keyword_list = data.get('keyword')
    
    if isinstance(keyword_list, list):
        keyword = ", ".join(keyword_list)
    else:
        keyword = keyword_list  
    
    # Query 벡터 생성
    query_vector = query_embed(keyword)

    search_params = {"metric_type": "IP", "params": {"ef": 64}}
    collections = ["festival_hereforus", "performance_hereforus", "food_hereforus"]
    aggregated_results = []
 
    # 컬렉션에서 검색
    for collection_name in collections:
        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
            collection.load()
            
            results = collection.search(
                data=[query_vector],
                anns_field="embedding",
                param=search_params,
                limit=1,  
                output_fields=["id", "text"]
            )

            aggregated_results.extend([
                {
                    "collection": collection_name,  # 컬렉션 이름 추가
                    "id": hit.entity.get("id"),
                    "distance": hit.distance,
                    "text": hit.entity.get("text")
                }
                for hit in results[0]
            ])

    # JSON 직렬화가 가능한 데이터 반환
    print(aggregated_results)
    return aggregated_results


@app.route('/course', methods=['POST'])
def recommendByClova():
    data = recommendByDB()

    # 컬렉션별로 데이터 분리
    festival_results = [item for item in data if item["collection"] == "festival_hereforus"]
    performance_results = [item for item in data if item["collection"] == "performance_hereforus"]
    food_results = [item for item in data if item["collection"] == "food_hereforus"]

    print(performance_results)

    # 축제와 공연 각각의 id 리스트 생성
    festival_ids = [item["id"] for item in festival_results]
    performance_ids = [item["id"] for item in performance_results]
    food_ids = [item["id"] for item in food_results]

    # 축제와 공연 각각의 text 리스트 생성
    festival_texts = [item["text"] for item in festival_results]
    performance_texts = [item["text"] for item in performance_results]
    food_texts = [item["text"] for item in food_results]

    # 요청 데이터 읽기
    bodyResponse = request.get_json()
    keyword_list = bodyResponse.get('keyword')
    
    completion_executor = CompletionExecutor(
        host=os.environ.get('CLOVASTUDIO_MODEL_HOST'),
        api_key=os.environ.get('CLOVASTUDIO_MODEL_API_KEY'),
        api_key_primary_val=os.environ.get('CLOVASTUDIO_MODEL_APIGW_API_KEY'),
        request_id=os.environ.get('CLOVASTUDIO_MODEL_REQUEST_ID'),
    )

    preset_text = [
    {
        "role": "system",
        "content": (
            f"- {keyword_list[0]}에 있는 {keyword_list[3]} 시간대부터 {keyword_list[1]}에서 놀 만한 {keyword_list[2]} 분위기의 {keyword_list[5]}이 있는 장소들과 {keyword_list[4]} 종류의 음식점 한 곳을 포함한 데이트 코스를 추천해줘.\n\n"
            "-  장소를 **시간대별로** 1시간 간격으로 추천해줘. 동선이 효율적이도록 고려해서 장소 간 이동이 효율적이도록 해줘. \n"
            "- 각 장소의 방문 이유와 예상 활동을 간단히 설명해주세요.\n"
            "- 시간대에 맞는 장소를 추천합니다. 특히 음식점은 한 번 정도만 추천하는 것이 적당합니다."
            "- 음식점과 공연, 축제에 대한 정보는 반드시 reference에 포함된 음식점 정보를 반드시 활용하여 추천합니다.\n"
            "- 답변 형식:\n"
            "  - 반드시 친근한 반말로 답변해줘.\n"
        )
    }
    ]

    preset_text.append(
        {
            "role": "assistant",
            "content": f"reference: {festival_texts}"
        }
    )

    preset_text.append(
        {
            "role": "assistant",
            "content": f"reference: {performance_texts}"
        }
    )

    
    preset_text.append(
        {
            "role": "assistant",
            "content": f"reference: {food_texts}"
        }
    )

    print(preset_text)

    request_data = {
        'messages': preset_text,
        'topP': 0.2,
        'topK': 0,
        'maxTokens': 1024,
        'temperature': 0.7,
        'repeatPenalty': 5.0,
        'stopBefore': [],
        'includeAiFilters': True,
        'seed': 0
    }
 
       # LLM 응답 생성
    try:
        response_data = completion_executor.execute(request_data)  # LLM 응답
        print(response_data)

        # JSON 응답 구성
        return jsonify({
            "llm_response": response_data,  # LLM의 추천 결과
            "festival_results": festival_results,  # 축제 데이터
            "performance_results": performance_results,  # 공연 데이터
            "food_results": food_results
        }), 200
    except Exception as e:
        print(f"Error during LLM response: {e}")
        return jsonify({"error": f"LLM 응답 중 오류 발생: {str(e)}"}), 500

    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT')))