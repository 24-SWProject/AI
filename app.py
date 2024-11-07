from flask import Flask, request, jsonify
from pymilvus import Collection, utility
from dataset.food import *
from dataset.festival import * 
# from dataset.weather import *
from dataset.clova import *
from dataset.performance import *
from dataset.weather import *
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

# weather data (cycle: every day)
@app.route('/weather', methods=['GET'])
def get_weather():
    try:
        indexing_weather_data()
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
    collections = ["festival_hereforus", "food_hereforus", "performance_hereforus"]  

    data = request.get_json()
    keyword = data.get('keyword')
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
                 "- `place` 항목은 반드시 5개의 장소를 고정으로 반환해야 합니다. "
                "장소들은 사용자의 키워드에 맞춰, 각기 다른 카테고리(예: 음식점, 카페, 볼거리 등)를 포함해야 합니다. "
                "각 장소는 사용자가 입력한 분위기, 음식 종류, 활동, 테마, 시간대 등의 키워드를 최대한 반영하여 추천합니다.\n\n"
                "- 사용자가 `행사`, `공연` 키워드를 입력할 경우, `place` 항목의 첫 번째 장소는 reference 내용을 바탕으로 한 행사 장소가 되어야 하며, 나머지 4개 장소는 AI가 다른 장소로 추천해야 합니다.\n\n"
                "- 모든 장소는 최적의 동선을 고려해 추천하며, JSON 형식과 필드명은 예시와 정확히 일치해야 합니다.\n\n"
                "JSON 응답 예시:\n\n"
                "{\n"
                "  \"place\": [\n"
                "    {\n"
                "      \"guName\": \"예시구\",\n"
                "      \"place\": \"예시아드레스\",\n"
                "      \"name\": \"예시이름\",\n"
                "      \"category\": \"예시카테고리\",\n"
                "      \"openDate\": \"공연, 행사 시작일자\",\n"
                "      \"endDate\": \"공연, 행사 종료일자\",\n"
                "      \"poster\": \"공연, 행사 이미지\",\n"
                "      \"link\": \공연, 행사 링크\",\n"
                "    }\n"
                "  ]\n"
                "}\n\n"
                "- JSON 형식과 필드명은 예시와 정확히 일치해야 하며, `place` 항목에는 항상 최소 5개의 장소를 포함해야 합니다."
            )
        }
        ]
    
    reference_content = "\n".join([f"{ref['text']}" for ref in reference])
    preset_text.append({
        "role": "system",
        "content": f"reference: {reference_content}"
    })
 
    preset_text.append({"role": "user", "content": keyword})

    # print(preset_text)

    request_data = {
        'messages': preset_text,
        'topP': 0.2,
        'topK': 0,
        'maxTokens': 1024,
        'temperature': 0.4,
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