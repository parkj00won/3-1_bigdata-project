import requests
import pandas as pd
import time

SERVICE_KEY = "5d705888c32914d2956f69efe4660568066f4c4d2eecec1a8717a13fe1c914ec"

BASE_URL = "https://apis.data.go.kr/1471000/CsmtcsIngdCpntInfoService01"


def fetch_api_to_dataframe(base_url, service_key, total_pages=10, rows_per_page=100):
    all_items = []

    for page in range(1, total_pages + 1):
        params = {
            "serviceKey": service_key,
            "pageNo": page,
            "numOfRows": rows_per_page,
            "type": "json"
        }

        response = requests.get(base_url, params=params)

        print(f"{page}페이지 요청 상태:", response.status_code)

        if response.status_code != 200:
            print("요청 실패")
            print(response.text[:500])
            continue

        data = response.json()

        print("응답 구조 확인:", data.keys())

        # 공공데이터 API마다 구조가 조금씩 다릅니다.
        # 아래 부분은 실제 응답 구조에 맞게 수정해야 합니다.
        try:
            items = data["body"]["items"]
        except KeyError:
            try:
                items = data["response"]["body"]["items"]["item"]
            except KeyError:
                print("items 위치를 찾지 못했습니다.")
                print(data)
                break

        if isinstance(items, dict):
            items = [items]

        if not items:
            print("더 이상 데이터가 없습니다.")
            break

        all_items.extend(items)

        time.sleep(0.2)

    df = pd.DataFrame(all_items)
    return df


df = fetch_api_to_dataframe(
    base_url=BASE_URL,
    service_key=SERVICE_KEY,
    total_pages=20,
    rows_per_page=100
)

print(df.head())
print(df.shape)

df.to_csv("data/raw/cosmetic_ingredients_raw.csv", index=False, encoding="utf-8-sig")

print("CSV 저장 완료!")