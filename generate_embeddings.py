# generate_embeddings.py
import json
import numpy as np
import google.generativeai as genai
import os
import re

# ★★★ API 키 설정 (secrets.toml에서 가져오기) ★★★
try:
    import toml
    secrets = toml.load(".streamlit/secrets.toml")
    API_KEY = secrets["GOOGLE_API_KEY"]
except:
    API_KEY = input("Google API Key 입력: ")

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
print("베리타스 엔진 - 임베딩 사전 생성 스크립트")
print("=" * 60)

# ======================================
# 법령 임베딩 생성
# ======================================
print("\n[1/2] 법령 임베딩 생성 중...")

if not os.path.exists("statutes_data.txt"):
    print("❌ statutes_data.txt 파일이 없습니다!")
    exit(1)

with open("statutes_data.txt", "r", encoding="utf-8") as f:
    content = f.read()

statute_items = []
statute_embeddings = []

parts = re.split(r"\s*---END OF STATUTE---\s*", content)
parts = [p.strip() for p in parts if p.strip()]

print(f"   총 {len(parts)}개 법령 조항 발견")

for i, p in enumerate(parts):
    if i % 10 == 0:
        print(f"   진행: {i+1}/{len(parts)} ({(i+1)/len(parts)*100:.1f}%)")
    
    emb = embed_text(p)
    if emb:
        statute_items.append({"rag_index": p, "raw_text": p})
        statute_embeddings.append(emb)

# 저장
np.save("statutes_embeddings.npy", np.array(statute_embeddings))
with open("statutes_items.json", "w", encoding="utf-8") as f:
    json.dump(statute_items, f, ensure_ascii=False, indent=2)

print(f"   ✅ 완료: {len(statute_items)}개 법령 저장")
print(f"   파일: statutes_embeddings.npy, statutes_items.json")

# ======================================
# 판례 임베딩 생성
# ======================================
print("\n[2/2] 판례 임베딩 생성 중...")

precedent_items = []
precedent_embeddings = []

if os.path.exists("precedents_data.jsonl"):
    with open("precedents_data.jsonl", "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    print(f"   총 {len(lines)}개 판례 발견")
    
    for i, line in enumerate(lines):
        if i % 10 == 0:
            print(f"   진행: {i+1}/{len(lines)} ({(i+1)/len(lines)*100:.1f}%)")
        
        try:
            obj = json.loads(line)
            txt = obj.get("rag_index") or obj.get("summary") or ""
            if not txt:
                continue
            
            emb = embed_text(txt)
            if emb:
                precedent_items.append(obj)
                precedent_embeddings.append(emb)
        except json.JSONDecodeError:
            continue
    
    # 저장
    np.save("precedents_embeddings.npy", np.array(precedent_embeddings))
    with open("precedents_items.json", "w", encoding="utf-8") as f:
        json.dump(precedent_items, f, ensure_ascii=False, indent=2)
    
    print(f"   ✅ 완료: {len(precedent_items)}개 판례 저장")
    print(f"   파일: precedents_embeddings.npy, precedents_items.json")

elif os.path.exists("precedents_data.txt"):
    with open("precedents_data.txt", "r", encoding="utf-8") as f:
        content = f.read()
    
    parts = re.split(r"\s*---END OF PRECEDENT---\s*", content)
    parts = [p.strip() for p in parts if p.strip()]
    
    print(f"   총 {len(parts)}개 판례 발견")
    
    for i, p in enumerate(parts):
        if i % 10 == 0:
            print(f"   진행: {i+1}/{len(parts)} ({(i+1)/len(parts)*100:.1f}%)")
        
        emb = embed_text(p)
        if emb:
            precedent_items.append({"rag_index": p, "raw_text": p})
            precedent_embeddings.append(emb)
    
    # 저장
    np.save("precedents_embeddings.npy", np.array(precedent_embeddings))
    with open("precedents_items.json", "w", encoding="utf-8") as f:
        json.dump(precedent_items, f, ensure_ascii=False, indent=2)
    
    print(f"   ✅ 완료: {len(precedent_items)}개 판례 저장")

else:
    print("   ⚠️ 판례 파일 없음 (precedents_data.jsonl 또는 .txt)")

print("\n" + "=" * 60)
print("🎉 임베딩 생성 완료!")
print("=" * 60)
print("\n생성된 파일:")
print("  - statutes_embeddings.npy")
print("  - statutes_items.json")
if precedent_items:
    print("  - precedents_embeddings.npy")
    print("  - precedents_items.json")
print("\n이제 app.py를 실행하면 0.5초 만에 로딩됩니다!")
