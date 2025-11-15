# precedents_loader_txt.py

"""
TXT 기반 판례 코퍼스 로더 (RAG용)

- 'precedents_data.txt' 파일만 사용
- 형식은 다음과 같이 저장되어 있다고 가정함:

==== PRECEDENT START ====
판례일련번호: 123456
사건번호: 2023도1234
선고일자: 2023.01.01
법원명: 대법원
사건종류명: 형사
판결유형: 판결
사건명: 도로교통법위반(음주운전)

[판시사항]
...

[판결요지]
...

[판례내용]
...
==== PRECEDENT END ====


이 모듈에서 제공하는 것:

- load_precedents_text(path): TXT 전체 로드
- split_precedents(raw_text): START/END 기준으로 한 건씩 분리
- load_corpus_for_rag(path): 최종적으로 RAG에 넣을 docs 리스트 반환
"""

import os
from typing import List

# 기본 TXT 경로
TXT_PATH = "precedents_data.txt"


def load_precedents_text(path: str = TXT_PATH) -> str:
    """
    판례 TXT 파일 전체를 하나의 문자열로 로드한다.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"판례 TXT 파일을 찾을 수 없습니다: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_precedents(raw_text: str) -> List[str]:
    """
    '==== PRECEDENT START ====' 과 '==== PRECEDENT END ====' 사이를
    한 건의 판례로 인식해서 리스트로 쪼갠다.
    """
    docs: List[str] = []

    chunks = raw_text.split("==== PRECEDENT START ====")

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        if "==== PRECEDENT END ====" in chunk:
            chunk = chunk.split("==== PRECEDENT END ====")[0].strip()

        if chunk:
            docs.append(chunk)

    return docs


def load_corpus_for_rag(path: str = TXT_PATH) -> List[str]:
    """
    RAG용 코퍼스를 불러오는 헬퍼 함수.
    """
    raw = load_precedents_text(path)
    docs = split_precedents(raw)
    return docs


if __name__ == "__main__":
    # 단독 실행 테스트용
    try:
        corpus = load_corpus_for_rag(TXT_PATH)
        print(f"[INFO] 총 {len(corpus)}건의 판례를 로드했습니다.")
        if corpus:
            print("\n[PREVIEW] 첫 번째 판례 일부 미리보기:\n")
            print(corpus[0][:2000])
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
