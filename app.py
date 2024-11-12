from flask import Flask, request, jsonify
from pymilvus import Collection, utility
from dataset.food import *
from dataset.festival import * 
# from dataset.weather import *
from dataset.clova import *
from dataset.performance import *
from dataset.movie import *
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

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
        indexing_food_data()
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

# recommend course
@app.route('/course', methods=['POST'])
def html_chat():
    connect_to_milvus()
    collection_name = "festival_hereforus"
    collections = ["festival_hereforus", "food_hereforus", "performance_hereforus", "movie_hereforus"]  

    data = request.get_json()
    keyword_list = data.get('keyword')
    
    if isinstance(keyword_list, list):
        keyword = ", ".join(keyword_list)
    else:
        keyword = keyword_list  # 이미 문자열이라면 그대로 사용
    
    query_vector = query_embed(keyword)
    
    search_params = {"metric_type": "IP", "params": {"ef": 64}}
    reference = []

    for collection_name in collections:
        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
            collection.load()
            
            results = collection.search(
                data=[query_vector],
                anns_field="embedding",
                param=search_params,
                limit=5,  
                output_fields=["text"]
            )
            
            # 검색 결과를 reference 리스트에 추가
            for hit in results[0]:
                reference.append({
                    "distance": hit.distance,
                    "text": hit.entity.get("text")
                })

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
            "- 음식점, 행사, 관광 명소 중 각 1개씩, 총 3개의 장소를 추천해야 합니다.\n\n"
            "- 음식점은 `reference`의 정보에 기반하여 추천하며, **행사는 영화, 공연, 축제 중 사용자가 입력한 키워드에 맞는 항목을 `reference`에서 추천**합니다.\n\n"
            "- 관광 명소는 사용자가 입력한 키워드(예: 분위기, 활동, 테마, 시간대 등)를 반영하여 AI가 추천합니다.\n\n"
            "- '영화' 키워드를 입력한 경우에는 `reference`에서 최신 개봉 영화 또는 높은 박스오피스 순위의 영화를 추천합니다.\n\n"
            "- 응답 형식:\n"
            "  예: '중구에서 저녁에 실내에서 조용히 즐길 수 있는 음식점과 영화, 명소들을 추천해 드리겠습니다.'\n\n"
            "  1. **음식점**: [음식점 이름] - 상세 설명만, 주소 제외\n\n"
            "  2. **행사**: [영화/공연/축제 이름] - 상세 설명만\n\n"
            "  3. **관광 명소**: [관광 명소 이름] - 상세 설명만\n\n"
            "- 각 장소에 대해 풍부한 설명을 제공하며, **주소나 위치 정보는 포함하지 않습니다.**\n\n"
            "- 반드시 음식점, 행사, 관광 명소 3가지를 포함한 추천을 제공합니다."
        )
    }
]


    
    reference_content = "\n".join([f"{ref['text']}" for ref in reference])
    preset_text.append({
        "role": "system",
        "content": f"reference: {reference_content}"
    })
 
    preset_text.append({"role": "user", "content": keyword})

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

 
    # LLM 생성 답변 반환
    try:
        response_data = completion_executor.execute(request_data)
        print(response_data)
        return jsonify({"response": response_data}), 200
    except Exception as e:
        print(f"Error during LLM response: {e}")
        return jsonify({"error": f"LLM 응답 중 오류 발생: {str(e)}"}), 500


    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT')))