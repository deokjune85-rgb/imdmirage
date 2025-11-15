# app.py

import os
from typing import List

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

from precedents_loader_txt import load_corpus_for_rag

# 1) 환경변수에서 OpenAI 키 읽기 (OPENAI_API_KEY 설정 필수)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("환경변수 OPENAI_API_KEY 를 먼저 설정하세요.")

client = OpenAI(api_key=OPENAI_API_KEY)

# 2) FastAPI 앱 생성
app = FastAPI(
    title="법률 판례 RAG 챗봇",
    description="precedents_data.txt 기반 판례 RAG 챗봇",
    version="1.0.0",
)

# 3) 전역 변수 (서버 시작 시 1번만 로딩)
DOCS: List[str] = []
EMB_MATRIX: np.ndarray | None = None


# 4) 임베딩 함수
def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    OpenAI text-embedding-3-small 을 사용해 리스트 임베딩
    """
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [d.embedding for d in resp.data]


# 5) 코사인 유사도 기반 검색
def search_docs(query: str, k: int = 5) -> List[str]:
    """
    쿼리와 가장 비슷한 판례 k개를 반환
    """
    global DOCS, EMB_MATRIX

    if EMB_MATRIX is None or len(DOCS) == 0:
        raise RuntimeError("코퍼스/임베딩이 로드되지 않았습니다.")

    # 쿼리 임베딩
    q_emb = np.array(embed_texts([query])[0], dtype=np.float32)  # (d,)

    # 문서 임베딩과 코사인 유사도 계산
    doc_embs = EMB_MATRIX  # (n, d)
    q_norm = np.linalg.norm(q_emb) + 1e-10
    doc_norms = np.linalg.norm(doc_embs, axis=1) + 1e-10

    scores = (doc_embs @ q_emb) / (doc_norms * q_norm)  # (n,)
    top_idx = np.argsort(-scores)[:k]

    return [DOCS[i] for i in top_idx]


# 6) LLM에게 최종 답변 생성 요청
def generate_answer(question: str, contexts: List[str]) -> str:
    """
    선택된 판례들을 '컨텍스트'로 넣고, 질문에 대한 답변 생성
    """
    context_text = ""
    for i, c in enumerate(contexts, start=1):
        context_text += f"[판례 {i}]\n{c}\n\n"

    system_prompt = (
        "너는 대한민국 변호사를 보조하는 법률 AI 어시스턴트다. "
        "아래에 제공된 판례들을 최대한 활용하여 사용자의 질문에 답해라. "
        "모르는 부분은 모른다고 말하고, 판례에 근거가 있는 부분만 자신 있게 말해라."
    )

    user_prompt = (
        f"다음은 판례들이다:\n\n{context_text}\n\n"
        f"위 판례들을 참고하여, 아래 질문에 한국어로 자세히 답변해줘.\n\n"
        f"[질문]\n{question}"
    )

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    return resp.choices[0].message.content


# 7) 요청 스키마
class Question(BaseModel):
    question: str
    top_k: int | None = 5


# 8) 서버 시작할 때 한 번만 코퍼스 + 임베딩 로딩
@app.on_event("startup")
def load_corpus_and_embeddings():
    global DOCS, EMB_MATRIX

    print("[INFO] precedents_data.txt 에서 코퍼스 로드 중...")
    DOCS = load_corpus_for_rag()  # TXT 전체 로딩 → 판례별 분리

    if not DOCS:
        raise RuntimeError("precedents_data.txt 에서 판례를 하나도 찾지 못했습니다.")

    print(f"[INFO] 총 {len(DOCS)} 건의 판례를 로드했습니다.")
    print("[INFO] 판례 임베딩 계산 중...(최초 1회)")

    embeddings = embed_texts(DOCS)
    EMB_MATRIX = np.array(embeddings, dtype=np.float32)

    print("[INFO] 임베딩 준비 완료. RAG 챗봇 준비 완료.")


# 9) 헬스체크
@app.get("/health")
def health_check():
    return {"status": "ok", "docs": len(DOCS)}


# 10) RAG 챗 엔드포인트
@app.post("/chat")
def chat(question: Question):
    """
    사용 예:
    POST /chat
    { "question": "혈중알콜농도 0.08인데 초범입니다. 실형 가능성?" }
    """
    top_k = question.top_k or 5

    # 1) 판례 검색 (음주운전이든 사기든, TXT에 들어있는 건 전부 검색 대상)
    contexts = search_docs(question.question, k=top_k)

    # 2) LLM으로 최종 답변 생성
    answer = generate_answer(question.question, contexts)

    return {
        "question": question.question,
        "top_k": top_k,
        "num_docs": len(contexts),
        "answer": answer,
        "contexts_preview": [c[:300] for c in contexts],  # 앞부분 미리보기
    }
