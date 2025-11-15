# app.py

from rag_loader import load_corpus_for_rag

# ============================
# 1) 앱 시작할 때 한 번만 판례 전부 로딩
# ============================
print("판례 TXT 로딩 중...")
CORPUS = load_corpus_for_rag()
print(f"총 {len(CORPUS)}건 로딩 완료.\n")


# ============================
# 2) 존나 단순한 검색기 (포함 여부만 체크)
#    - 나중에 여기만 벡터스토어/임베딩으로 바꾸면 됨
# ============================
def search_precedents(query: str, top_k: int = 5):
    """
    초간단 버전:
    - query 문자열이 들어간 판례만 골라서 위에서부터 top_k개 잘라서 리턴
    """
    results = []
    for doc in CORPUS:
        if query in doc:
            results.append(doc)
        if len(results) >= top_k:
            break
    return results


# ============================
# 3) 터미널에서 테스트용 REPL
# ============================
if __name__ == "__main__":
    print("판례 검색 테스트 모드입니다.")
    print("예: 음주운전 / 유사수신 / 사기 이런 식으로 쳐봐.")
    print("그만하려면 그냥 엔터 두 번.\n")

    while True:
        q = input("질문(키워드): ").strip()
        if not q:
            break

        hits = search_precedents(q, top_k=3)

        if not hits:
            print("\n[결과 없음]\n")
            continue

        print("\n================ 검색 결과 ================\n")
        for i, doc in enumerate(hits, 1):
            print(f"[{i}] ----------------------------------------")
            # 너무 기니까 앞부분만 맛보기
            print(doc[:1500])
            print("\n(생략)\n")
        print("===========================================\n")
