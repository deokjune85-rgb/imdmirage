import os
from typing import List, Tuple

from openai import OpenAI

from rag_loader import load_corpus_for_rag

# ========= 1. 전역 설정 =========

# OpenAI 클라이언트 (환경변수 OPENAI_API_KEY 필요)
client = OpenAI()

# RAG용 말뭉치 로드 (precedents_data.txt 기반)
try:
    CORPUS: List[str] = load_corpus_for_rag()
    print(f"[INFO] 판례/법령 문서 로드 완료. 총 {len(CORPUS)}건")
except Exception as e:
    print(f"[ERROR] 판례 TXT 로드 실패: {e}")
    CORPUS = []


# ========= 2. 단순 검색 함수 =========

def simple_score(query: str, doc: str) -> int:
    """
    아주 단순한 키워드 기반 스코어링.
    나중에 벡터 DB 붙이면 여기 부분만 갈아끼우면 됨.
    """
    q = query.lower()
    d = doc.lower()
    score = 0
    # 공백 기준으로 쪼개서 키워드별 개수 세기
    for token in q.split():
        token = token.strip()
        if not token:
            continue
        score += d.count(token)
    return score


def search_precedents(query: str, top_k: int = 5) -> List[Tuple[int, str]]:
    """
    CORPUS에서 query와 가장 관련 있어 보이는 상위 top_k 개 문서 반환.
    [(점수, 문서내용), ...] 형태.
    """
    if not CORPUS:
        return []

    scored = []
    for doc in CORPUS:
        s = simple_score(query, doc)
        if s > 0:
            scored.append((s, doc))

    if not scored:
        return []

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


# ========= 3. 프롬프트 구성 =========

def build_system_prompt() -> str:
    return (
        "너는 대한민국 판례 / 법령 RAG 엔진의 법률 분석 파트너다. "
        "아래에 제공되는 판례/법령 발췌문을 최대한 활용해서 사용자의 질문에 답변한다.\n\n"
        "- 판례 / 법령 요지는 가능한 한 사실 위주로 정리한다.\n"
        "- 숫자(조문 번호, 사건번호 등)는 문맥상 확실할 때만 사용한다.\n"
        "- 모호할 경우 '이 자료만으로는 단정하기 어렵다'고 분명히 말한다.\n"
        "- 변호사 대리 없이는 위험할 수 있는 사안이면 반드시 '변호사 선임 필요성'을 안내한다.\n"
    )


def build_user_prompt(query: str, contexts: List[str]) -> str:
    context_block = ""
    for i, ctx in enumerate(contexts, start=1):
        context_block += f"\n\n[자료 {i}]\n{ctx}\n"

    user_prompt = (
        "다음은 대한민국 판례·법령 RAG에서 찾아온 관련 자료들이다."
        " 이 자료를 중심으로 아래 질문에 답해라.\n\n"
        f"=== RAG 자료 시작 ===\n{context_block}\n=== RAG 자료 끝 ===\n\n"
        f"[질문]\n{query}\n\n"
        "[요청]\n"
        "1) 판례/법령에서 직접 읽히는 내용 위주로 사실관계와 법리를 정리하고,\n"
        "2) 질문자 사건에 대략 어떻게 적용될 수 있을지 '가능성' 수준에서만 말하며,\n"
        "3) 실제 대응 전략(수사 단계/재판 단계에서 어떤 자료를 준비해야 하는지 등)을 제안하라.\n"
    )
    return user_prompt


# ========= 4. LLM 호출 =========

def answer_with_rag(query: str) -> str:
    # 1) CORPUS에서 관련 문서 찾기
    results = search_precedents(query, top_k=5)
    contexts = [doc for _, doc in results]

    # 2) 콘텍스트가 없으면 그냥 일반 답변
    if not contexts:
        print("[WARN] 검색된 판례/법령 자료가 없습니다. 일반 답변 모드로 처리합니다.")
        messages = [
            {
                "role": "system",
                "content": (
                    "너는 대한민국 형사/민사/행정 등 전반에 대해 개괄적인 설명을 해주는 "
                    "법률 안내 도우미다. RAG 자료가 없는 상태이므로, 일반적인 법률 지식을 기준으로만 답변해라. "
                    "구체적인 사건에 대한 최종 판단은 반드시 변호사 상담이 필요하다고 안내해라."
                ),
            },
            {"role": "user", "content": query},
        ]
    else:
        print(f"[INFO] RAG 문서 {len(contexts)}건 사용")
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(query, contexts)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    # 3) OpenAI LLM 호출
    response = client.chat.completions.create(
        model="gpt-4.1-mini",  # 필요하면 다른 모델로 변경
        messages=messages,
        temperature=0.1,
    )

    return response.choices[0].message.content.strip()


# ========= 5. CLI 진입점 =========

def main():
    print("=== VERITAS RAG 판례·법령 엔진 (CLI 버전) ===")
    print("종료하려면 'q' 또는 'quit' 입력\n")

    while True:
        try:
            query = input("\n질문 입력 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[종료]")
            break

        if query.lower() in ("q", "quit", "exit"):
            print("[종료]")
            break

        if not query:
            continue

        try:
            answer = answer_with_rag(query)
            print("\n[답변]\n")
            print(answer)
        except Exception as e:
            print(f"\n[ERROR] 답변 생성 중 오류: {e}")


if __name__ == "__main__":
    main()
