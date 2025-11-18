# generate_embeddings.py
import json
import numpy as np
import google.generativeai as genai
import os
import re

# API 키
try:
    import toml
    secrets = toml.load(".streamlit/secrets.toml")
    API_KEY = secrets["GOOGLE_API_KEY"]
except:
    API_KEY = input("Google API Key: ")

genai.configure(api_key=API_KEY)

EMBEDDING_MODEL = "models/text-embedding-004"

def embed_text(text):
    clean_text = text.replace("\n", " ").strip()
    if not clean_text:
        return None
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=clean_text,
            task_type="retrieval_document",
        )
        return result["embedding"]
    except Exception as e:
        print(f"[Error] {e}")
        return None

print("=" * 60)
print("임베딩 생성 시작")
print("=" * 60)

# 1. 법령 임베딩
print("\n[1/2] 법령 임베딩...")
statute_items = []
statute_embeddings = []

if os.path.exists("statutes_data.txt"):
    with open("statutes_data.txt", "r", encoding="utf-8") as f:
        content = f.read()
    
    parts = re.split(r"\s*---END OF STATUTE---\s*", content)
    parts = [p.strip() for p in parts if p.strip()]
    
    print(f"   총 {len(parts)}개 법령 발견")
    
    for i, p in enumerate(parts):
        if i % 5 == 0:
            print(f"   진행: {i+1}/{len(parts)}")
        
        emb = embed_text(p)
        if emb:
            statute_items.append({"rag_index": p, "raw_text": p})
            statute_embeddings.append(emb)
    
    np.save("statutes_embeddings.npy", np.array(statute_embeddings))
    with open("statutes_items.json", "w", encoding="utf-8") as f:
        json.dump(statute_items, f, ensure_ascii=False, indent=2)
    
    print(f"   ✅ 완료: {len(statute_items)}개")
else:
    print("   ❌ statutes_data.txt 없음!")

# 2. 판례 임베딩 (임시 생성)
print("\n[2/2] 판례 생성 및 임베딩...")

precedents = [
    {
        "id": "2023도12345",
        "title": "마약류관리법위반(향정)-매매",
        "court": "대법원",
        "date": "2023-05-15",
        "summary": "필로폰 판매 공범 실형 3년",
        "rag_index": "피고인이 공동으로 필로폰 50g을 판매한 사실이 인정됨. 조직적 유통망에 관여하였고 영리 목적이 명백하여 징역 3년 실형 선고",
        "full_text": "피고인들은 2023년 3월부터 5월까지 공동으로 필로폰 50g을 판매하였다. 조직적 유통망에 관여하였고 영리 목적이 명백하다. 징역 3년을 선고한다.",
        "url": "https://example.com/case1"
    },
    {
        "id": "2022도67890",
        "title": "마약류관리법위반(향정)-매매",
        "court": "서울중앙지법",
        "date": "2022-11-20",
        "summary": "필로폰 판매 초범 집행유예",
        "rag_index": "피고인이 필로폰 10g을 판매하였으나 초범이고 자백하며 깊이 반성하여 징역 2년 집행유예 3년 선고",
        "full_text": "피고인은 필로폰 10g을 판매하였다. 초범이고 자백하며 반성하는 태도를 보였다. 징역 2년 집행유예 3년을 선고한다.",
        "url": "https://example.com/case2"
    },
    {
        "id": "2023도11111",
        "title": "마약류관리법위반(향정)-투약/소지",
        "court": "수원지법",
        "date": "2023-08-10",
        "summary": "필로폰 투약 및 소지 실형 1년 6월",
        "rag_index": "피고인이 필로폀을 투약하고 5g을 소지한 사실이 인정됨. 동종 전과 1회 있어 징역 1년 6월 실형 선고",
        "full_text": "피고인은 필로폰을 투약하고 5g을 소지하였다. 동종 전과가 있어 징역 1년 6월을 선고한다.",
        "url": "https://example.com/case3"
    },
    {
        "id": "2023도22222",
        "title": "마약류관리법위반(향정)-매매/알선",
        "court": "대법원",
        "date": "2023-09-25",
        "summary": "마약 알선 및 판매 실형 4년",
        "rag_index": "피고인이 마약 거래를 알선하고 직접 판매도 병행한 사실이 인정됨. 조직적 범행으로 징역 4년 실형 선고",
        "full_text": "피고인은 마약 거래를 알선하고 직접 판매도 하였다. 조직적 범행으로 징역 4년을 선고한다.",
        "url": "https://example.com/case4"
    },
    {
        "id": "2022도33333",
        "title": "마약류관리법위반(향정)-매매",
        "court": "부산지법",
        "date": "2022-12-15",
        "summary": "필로폰 대량 판매 실형 5년",
        "rag_index": "피고인이 필로폰 200g을 판매하여 대규모 유통에 관여한 사실이 인정됨. 징역 5년 실형 선고",
        "full_text": "피고인은 필로폰 200g을 판매하였다. 대규모 유통에 관여하여 징역 5년을 선고한다.",
        "url": "https://example.com/case5"
    }
]

# JSONL 저장
with open("precedents_data.jsonl", "w", encoding="utf-8") as f:
    for p in precedents:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

print(f"   판례 {len(precedents)}개 생성")

# 임베딩
precedent_items = []
precedent_embeddings = []

for i, p in enumerate(precedents):
    print(f"   임베딩: {i+1}/{len(precedents)}")
    emb = embed_text(p["rag_index"])
    if emb:
        precedent_items.append(p)
        precedent_embeddings.append(emb)

np.save("precedents_embeddings.npy", np.array(precedent_embeddings))
with open("precedents_items.json", "w", encoding="utf-8") as f:
    json.dump(precedent_items, f, ensure_ascii=False, indent=2)

print(f"   ✅ 완료: {len(precedent_items)}개")

print("\n" + "=" * 60)
print("🎉 완료!")
print("=" * 60)
