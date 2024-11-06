from flask import Flask, request, jsonify
from pymilvus import Collection, utility
from dataset.food import *
from dataset.festival import * 
from dataset.weather import *
from dataset.clova import *
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

# recommend course
@app.route('/course', methods=['POST'])
def html_chat():
    connect_to_milvus()
    collection_name = "festival_hereforus"
    collections = ["festival_hereforus", "food_hereforus"]  

    data = request.get_json()
    realquery = data.get('realquery')
    query_vector = query_embed(realquery)

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
        "content": "- 답변할 때 반드시 주어진 reference의 내용만을 바탕으로 답변하세요. 다만, `food` 항목은 reference가 없는 경우에도 기본 데이터를 제공해야 합니다.\n\n"
                "- `festival` 항목에는 반드시 하나의 항목만 포함하고, `food` 항목에는 일반음식점, 제과제빵점, 휴게음식점의 각기 다른 카테고리에서 한 개씩 반환해야 합니다.\n\n"
                "JSON 응답 예시:\n\n"
                "{\n"
                "  \"festival\": [\n"
                "    {\n"
                "      \"guName\": \"예시구\",\n"
                "      \"title\": \"예시제목\",\n"
                "      \"place\": \"예시장소\",\n"
                "      \"target\": \"예시대상\",\n"
                "      \"link\": \"예시링크\",\n"
                "      \"imageUrl\": \"예시이미지링크\",\n"
                "      \"dates\": {\n"
                "        \"start\": \"시작날짜\",\n"
                "        \"end\": \"종료날짜\"\n"
                "      },\n"
                "      \"coordinates\": {\n"
                "        \"x\": \"x좌표\",\n"
                "        \"y\": \"y좌표\"\n"
                "      }\n"
                "    }\n"
                "  ],\n"
                "  \"food\": [\n"
                "    {\n"
                "      \"id\": 0,\n"
                "      \"guName\": \"예시구\",\n"
                "      \"address\": \"예시아드레스\",\n"
                "      \"name\": \"예시음식점\",\n"
                "      \"category\": \"일반음식점\",\n"
                "      \"coordinates\": {\n"
                "        \"x\": \"x좌표\",\n"
                "        \"y\": \"y좌표\"\n"
                "      }\n"
                "    },\n"
                "    {\n"
                "      \"id\": 1,\n"
                "      \"guName\": \"예시구\",\n"
                "      \"address\": \"예시아드레스\",\n"
                "      \"name\": \"예시베이커리\",\n"
                "      \"category\": \"제과제빵점\",\n"
                "      \"coordinates\": {\n"
                "        \"x\": \"x좌표\",\n"
                "        \"y\": \"y좌표\"\n"
                "      }\n"
                "    },\n"
                "    {\n"
                "      \"id\": 2,\n"
                "      \"guName\": \"예시구\",\n"
                "      \"address\": \"예시아드레스\",\n"
                "      \"name\": \"예시카페\",\n"
                "      \"category\": \"휴게음식점\",\n"
                "      \"coordinates\": {\n"
                "        \"x\": \"x좌표\",\n"
                "        \"y\": \"y좌표\"\n"
                "      }\n"
                "    }\n"
                "  ]\n"
                "}\n\n"
                "- JSON 형식과 필드명은 예시와 정확히 일치해야 하며, `festival`에는 반드시 하나의 항목만 포함하고, `food`에는 각기 다른 세 가지 카테고리의 항목을 반드시 포함해야 합니다."
    }
    ]
    
    reference_content = "\n".join([f"{ref['text']}" for ref in reference])
    preset_text.append({
        "role": "system",
        "content": f"reference: {reference_content}"
    })
 
    preset_text.append({"role": "user", "content": realquery})

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
    app.run(debug=True, host='0.0.0.0', port=os.environ.get('PORT'))