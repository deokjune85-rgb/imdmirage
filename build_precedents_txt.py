import requests
import xml.etree.ElementTree as ET
import time

# ==============================
# 설정값
# ==============================
BASE_URL = "https://www.law.go.kr"
OC = "deokjune"  # 네가 쓰는 law.go.kr 아이디 앞부분

# 결과를 저장할 TXT 파일 경로
OUTPUT_PATH = "precedents_data.txt"

# 👉 여기 키워드 리스트에다가 원하는 죄명/키워드 계속 추가하면 됨
KEYWORDS = [
    "음주운전",
    "사기",
    "절도",
    "폭행",
    "상해",
    "특수상해",
    "특가법",
    "성폭력",
    "성매매",
    "횡령",
    "배임",
    "도박",
    "마약",
    "유사수신",
    "공갈",
    "협박",
    "강도",
    "살인",
]


# ==============================
# 판례 목록 검색 (lawSearch.do)
# ==============================
def search_prec_ids_by_keyword(keyword: str, max_pages: int = 5, display: int = 20):
    """
    특정 키워드로 판례 검색해서 '판례일련번호' 목록만 뽑아오는 함수
    """
    all_ids = []
    for page in range(1, max_pages + 1):
        params = {
            "OC": OC,
            "target": "prec",
            "type": "XML",
            "query": keyword,
            "page": page,
            "display": display,
        }
        url = f"{BASE_URL}/DRF/lawSearch.do"
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()

        root = ET.fromstring(resp.content)

        # 전체 건수 (필요하면 사용)
        total_cnt_text = root.findtext("totalCnt", default="0") or "0"
        try:
            total_cnt = int(total_cnt_text)
        except ValueError:
            total_cnt = 0

        for prec in root.findall("prec"):
            pid = prec.findtext("판례일련번호")
            if pid and pid not in all_ids:
                all_ids.append(pid)

        # 더 이상 페이지 없으면 끊기 (대략)
        if page * display >= total_cnt:
            break

        time.sleep(0.2)  # 너무 빡세게 안 두들기게 살짝 딜레이

    return all_ids


# ==============================
# 판례 상세 조회 (lawService.do)
# ==============================
def fetch_prec_detail(prec_id: str) -> dict:
    """
    lawService.do 로 판례 상세 내용 받아오기
    """
    params = {
        "OC": OC,
        "target": "prec",
        "type": "XML",
        "ID": prec_id,
    }
    url = f"{BASE_URL}/DRF/lawService.do"
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()

    root = ET.fromstring(resp.content)

    def get(tag: str) -> str:
        el = root.find(tag)
        if el is None or el.text is None:
            return ""
        return el.text.strip()

    data = {
        "판례일련번호": prec_id,
        "사건명": get("사건명"),
        "사건번호": get("사건번호"),
        "선고일자": get("선고일자"),
        "법원명": get("법원명"),
        "사건종류명": get("사건종류명"),
        "판결유형": get("판결유형"),
        "판시사항": get("판시사항"),
        "판결요지": get("판결요지"),
        "참조조문": get("참조조문"),
        "참조판례": get("참조판례"),
        "판례내용": get("판례내용"),
    }
    return data


# ==============================
# TXT에 들어갈 한 건 포맷팅
# ==============================
def make_precedent_block(kw: str, data: dict) -> str:
    """
    RAG에서 쪼갤 수 있게
    ==== PRECEDENT START ==== / ==== PRECEDENT END ====
    형태로 한 건씩 뭉쳐서 문자열 만들어줌
    """
    lines = []
    lines.append("==== PRECEDENT START ====")
    lines.append(f"[검색키워드] {kw}")
    lines.append(f"[판례일련번호] {data.get('판례일련번호', '')}")
    lines.append(f"[사건명] {data.get('사건명', '')}")
    lines.append(f"[사건번호] {data.get('사건번호', '')}")
    lines.append(f"[선고일자] {data.get('선고일자', '')}")
    lines.append(f"[법원명] {data.get('법원명', '')}")
    lines.append(f"[사건종류명] {data.get('사건종류명', '')}")
    lines.append(f"[판결유형] {data.get('판결유형', '')}")
    lines.append("")

    # 긴 텍스트들
    for field in ("판시사항", "판결요지", "참조조문", "참조판례", "판례내용"):
        value = data.get(field, "")
        if value:
            lines.append(f"[{field}]")
            lines.append(value)
            lines.append("")

    lines.append("==== PRECEDENT END ====")
    return "\n".join(lines)


# ==============================
# 전체 실행: 키워드 돌면서 TXT 생성
# ==============================
def build_precedents_txt():
    seen_ids = set()  # 중복 방지

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for kw in KEYWORDS:
            print(f"\n[*] 키워드 '{kw}' 판례 수집 중...")
            try:
                ids = search_prec_ids_by_keyword(kw, max_pages=5, display=20)
            except Exception as e:
                print(f"[!] 키워드 '{kw}' 검색 실패: {e}")
                continue

            print(f"    - 검색된 판례일련번호 개수: {len(ids)}")

            for pid in ids:
                if pid in seen_ids:
                    continue
                seen_ids.add(pid)

                try:
                    detail = fetch_prec_detail(pid)
                except Exception as e:
                    print(f"[!] 판례 {pid} 상세 조회 실패: {e}")
                    continue

                block = make_precedent_block(kw, detail)
                f.write(block + "\n\n")

                # 너무 과도한 호출 방지
                time.sleep(0.2)

    print(f"\n✅ 완료: {OUTPUT_PATH} 파일 생성됨")


if __name__ == "__main__":
    build_precedents_txt()
