# rag_loader.py

import os

# 판례/법령을 모아둔 TXT 파일 경로
TXT_PATH = "precedents_data.txt"


def load_precedents_text(path: str = TXT_PATH) -> str:
    """
    precedents_data.txt 전체 내용을 통으로 읽어오는 함수
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"판례 TXT 파일을 찾을 수 없습니다: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_precedents(raw_text: str):
    """
    TXT 저장 포맷 기준으로 한 건씩 쪼갠다.

    각 판례/법령 블록을 다음 형식으로 저장했다고 가정:
    ==== PRECEDENT START ====
    (여기에 한 건 전체 내용)
    ==== PRECEDENT END ====
    """
    docs = []
    chunks = raw_text.split("==== PRECEDENT START ====")

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        # 끝 표시 기준으로 자르기
        if "==== PRECEDENT END ====" in chunk:
            chunk = chunk.split("==== PRECEDENT END ====")[0].strip()

        if chunk:
            docs.append(chunk)

    return docs


def load_corpus_for_rag():
    """
    RAG에서 쓸 말뭉치 리스트를 반환한다.
    나머지는 app.py에서 CORPUS = load_corpus_for_rag() 로 써먹으면 됨.
    """
    raw = load_precedents_text()
    docs = split_precedents(raw)
    return docs
