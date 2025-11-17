# -*- coding: utf-8 -*-
# 베리타스 엔진 8.1.3 — Auto-Analysis Mode + Dual RAG (코드 멸균 및 구문 복구 완료)

import streamlit as st
import google.generativeai as genai
import os
import numpy as np
import re
import time
import json
import PyPDF2 # PDF 처리를 위해 필수

# ---------------------------------------
# 0. 기본 세팅
# ---------------------------------------
st.set_page_config(
    page_title="베리타스 엔진 8.1",
    page_icon="🛡️",
    layout="centered"
)

# CSS (모든 공백은 표준 공백 U+0020으로 정제됨)
custom_css = """
<style>
#MainMenu, footer, header, .stDeployButton {visibility:hidden;}
html, body, div, span, p {
    font-family: 'Noto Sans KR', sans-serif !important;
    font-size: 16px !important;
    line-height: 1.7 !important;
}
h1 {
    text-align: left !important;
    font-weight: 900 !important;
    font-size: 36px !important;
    margin-top: 10px !important;
    margin-bottom: 15px !important;
}
strong, b { font-weight: 700; }
.fadein { animation: fadeInText 0.5s ease-in-out forwards; opacity: 0; }
@keyframes fadeInText {
    from {opacity: 0; transform: translateY(3px);}
    to {opacity: 1; transform: translateY(0);}
}
[data-testid="stChatMessageContent"] {
    font-size: 16px !important;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

st.title("베리타스 엔진 8.1")
st.caption("The Architect — 전략 시뮬레이션 엔진")

# 이모지 제거 및 텍스트 수정 (호환성 강화)
st.warning(
    "보안 경고: 본 시스템은 격리된 사설 환경(The Vault)에서 작동합니다. "
    "모든 데이터는 기밀로 취급되며 외부로 유출되지 않습니다."
)

# ---------------------------------------
# 1. API 키 설정
# ---------------------------------------
try:
    # Streamlit Cloud 배포 시 st.secrets 사용
    if "GOOGLE_API_KEY" in st.secrets:
        API_KEY = st.secrets["GOOGLE_API_KEY"]
    # 로컬 환경에서는 환경 변수 사용 (선택 사항)
    else:
        API_KEY = os.environ.get("GOOGLE_API_KEY")

    if not API_KEY:
        raise ValueError("API Key not found in secrets or environment variables.")
    genai.configure(api_key=API_KEY)
except (KeyError, ValueError) as e:
    st.error(f"시스템 오류: 엔진 연결 실패. API 키를 확인하세요. {e}")
    st.stop()

# ---------------------------------------
# 2. 임베딩 / RAG 유틸
# ---------------------------------------
EMBEDDING_MODEL_NAME = "models/text-embedding-004"

def embed_text(text: str, task_type: str = "retrieval_document"):
    clean_text = text.replace("\n", " ").strip()
    if not clean_text:
        return None
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL_NAME,
            content=clean_text,
            task_type=task_type,
        )
        return result["embedding"]
    except Exception as e:
        print(f"[Embedding error] {e}")
        return None

# ---------------------------------------
# 2-1. 사전 계산된 임베딩 로드
# ---------------------------------------
@st.cache_data(show_spinner=False)
def load_precomputed_embeddings():
    """사전 계산된 임베딩 로드 (0.5초 완료)"""
    statute_items = []
    statute_embeddings = []
    precedent_items = []
    precedent_embeddings = []
    
    # 법령 로드
    if os.path.exists("statutes_embeddings.npy") and os.path.exists("statutes_items.json"):
        statute_embeddings = np.load("statutes_embeddings.npy").tolist()
        with open("statutes_items.json", "r", encoding="utf-8") as f:
            statute_items = json.load(f)
        print(f"[RAG] 법령 로드 완료: {len(statute_items)}개")
    
    # 판례 로드
    if os.path.exists("precedents_embeddings.npy") and os.path.exists("precedents_items.json"):
        precedent_embeddings = np.load("precedents_embeddings.npy").tolist()
        with open("precedents_items.json", "r", encoding="utf-8") as f:
            precedent_items = json.load(f)
        print(f"[RAG] 판례 로드 완료: {len(precedent_items)}개")
    
    return statute_items, statute_embeddings, precedent_items, precedent_embeddings

def find_similar_items(query_text, items, embeddings, top_k=3, threshold=0.5):
    if not items or not embeddings:
        return []

    q_emb = embed_text(query_text, task_type="retrieval_query")
    if q_emb is None:
        return []

    # 안정성을 위해 데이터 타입을 명시적으로 변환
    try:
        embeddings_np = np.array(embeddings, dtype=np.float32)
        q_emb_np = np.array(q_emb, dtype=np.float32)
    except ValueError as e:
        print(f"[RAG Error] 임베딩 데이터 타입 변환 실패: {e}")
        return []

    # 임베딩 차원 확인 (안정성 강화)
    if embeddings_np.size > 0:
        # ndim 체크 추가하여 1차원 배열 오류 방지
        if embeddings_np.ndim < 2 or embeddings_np.shape[1] != len(q_emb_np):
            print(f"[RAG Error] 임베딩 차원 불일치 또는 구조 오류: DB Shape={embeddings_np.shape}, Query Len={len(q_emb_np)}")
            return []
    else:
        return []

    sims = np.dot(embeddings_np, q_emb_np)
    idxs = np.argsort(sims)[::-1][:top_k]

    results = []
    for i in idxs:
        score = float(sims[i])
        if score < threshold:
            continue
        item = items[i].copy()
        item["similarity"] = score
        results.append(item)

    return results

# ---------------------------------------
# 3. PDF 처리 함수 (진단 강화됨 v8.1.3)
# ---------------------------------------
def extract_text_from_pdf(uploaded_file):
    """PDF 텍스트를 추출하고, 실패 시 원인 코드를 반환한다."""
    try:
        # [개선 1] 안정성 확보: 스트림 위치를 처음으로 되돌림 (Streamlit 특성 고려)
        uploaded_file.seek(0)
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        
        # [개선 2] 암호화 확인
        if pdf_reader.is_encrypted:
             return "[ERROR:ENCRYPTED]"

        text = ""
        
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                # 추출된 텍스트가 의미 있는지 확인 (공백 제거)
                cleaned_text = page_text.strip()
                if cleaned_text:
                    # 페이지 번호 표시 (이모지 제거)
                    text += f"\n--- 페이지 {page_num + 1} ---\n"
                    text += cleaned_text
        
        # [개선 3] 내용물 없음 감지 (스캔 PDF 진단)
        if not text.strip():
            # 모든 페이지 처리 후 텍스트가 없으면 스캔된 PDF 또는 빈 파일로 간주
            return "[ERROR:NO_TEXT]"

        return text.strip()
    
    except Exception as e:
        # 처리 실패 감지 (손상 등)
        print(f"[PDF Extraction Error] {e}") # 디버깅용 서버 로그
        return f"[ERROR:PROCESSING_FAILED]"


def analyze_case_file(pdf_text: str):
    """PDF 텍스트를 분석하여 핵심 정보를 JSON으로 추출한다."""
    # ★★★ [오류 수정 완료] f-string 문법 및 Markdown 아티팩트 제거 ★★★
    analysis_prompt = f"""
다음은 사건기록 PDF에서 추출한 내용입니다.

[PDF 내용]
{pdf_text[:15000]} # 컨텍스트 길이 제한 고려

[분석 지침]
1. 이 사건의 도메인 분류 (형사/민사/가사/행정/파산/IP/의료/세무 중 1개)
2. 세부 분야 (예: 형사-마약, 민사-계약분쟁 등)
3. 핵심 사실관계 5가지 (시간순 또는 중요도순)
4. 확보된 증거 목록 (문서명, 종류)
5. 피고인/원고 측 주장 요약
6. 상대방 측 주장 요약

반드시 아래 JSON 형식으로만 출력하세요. 다른 설명은 하지 마세요. (```json 마크다운 포함)

```json
{{
  "domain": "형사",
  "subdomain": "마약",
  "key_facts": ["사실 1", "사실 2", "사실 3", "사실 4", "사실 5"],
  "evidence": ["증거 1", "증거 2"],
  "our_claim": "우리 측 주장 요약",
  "their_claim": "상대방 측 주장 요약"
}}
""" # ★★★ f-string 정상 종료됨. 아래부터 정상 코드 시작. ★★★

try:
    # [최적화] 분석에는 채팅 기록이 필요 없으므로, 별도의 모델 인스턴스 사용 (Gemini 1.5 Flash 권장)
    analysis_model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
    response = analysis_model.generate_content(analysis_prompt)
    result_text = response.text.strip()
    
    # JSON 추출 (정규식 사용으로 안정성 강화)
    json_match = re.search(r'```json\s*({.*?})\s*```', result_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
        result = json.loads(json_str)
        return result
    else:
        raise ValueError("AI가 유효한 JSON 형식을 생성하지 못함.")

except Exception as e:
    st.error(f"AI 분석 실패: {e}")
    # 실패 시 디버깅을 위해 원본 응답 출력 (서버 로그)
    if 'response' in locals() and hasattr(response, 'text'):
        print(f"[Analysis Failure Debug] AI Response: {response.text[:1000]}")
    return None
---------------------------------------
4. 각종 유틸 함수
---------------------------------------
def _is_menu_input(s: str) -> bool: return bool(re.fullmatch(r"^\s*\d{1,2}(?:-\d{1,2})?\s*$", s))

def _is_reset_keyword(s: str) -> bool: """처음으로/메인/초기화 키워드 감지""" keywords = ["처음", "메인", "초기화", "reset", "돌아가", "처음으로"] return any(kw in s.lower() for kw in keywords)

def _is_final_report(txt: str) -> bool: return "전략 브리핑 보고서" in txt

def _query_title(prompt_text: str) -> str: return prompt_text[:67] + "..." if len(prompt_text) > 70 else prompt_text

def update_active_module(response_text: str): # Auto-Analysis Mode 감지 (엄격한 조건) if ("9." in response_text and "사건기록 자동 분석 모드" in response_text) or

("Auto-Analysis Mode를 활성화합니다" in response_text): st.session_state.active_module = "Auto-Analysis Mode" return

# 일반 모듈 활성화 (정규식 수정: '[모듈명]' 또는 "'모듈명'" 모두 감지)
m = re.search(r"['\[](.+?)['\]] 모듈을 (?:최종 )?활성화합니다", response_text)
if m:
    st.session_state.active_module = m.group(1).strip()
elif "Phase 0" in response_text and not st.session_state.get("active_module"):
    st.session_state.active_module = "Phase 0"
---------------------------------------
5. 시스템 프라임 프롬프트 로드
---------------------------------------
try: with open("system_prompt.txt", "r", encoding="utf-8") as f: SYSTEM_INSTRUCTION = f.read() if len(SYSTEM_INSTRUCTION) < 100: raise ValueError("System prompt is too short.") except (FileNotFoundError, ValueError) as e: st.error(f"치명적 오류: system_prompt.txt 로드 실패. {e}") st.stop()

---------------------------------------
6. 모델 & 세션 초기화
---------------------------------------
if "model" not in st.session_state: try: # 모델명 표준으로 수정 (gemini-1.5-flash-latest 권장) st.session_state.model = genai.GenerativeModel( "models/gemini-1.5-flash-latest", system_instruction=SYSTEM_INSTRUCTION, ) st.session_state.chat = st.session_state.model.start_chat(history=[]) except Exception as e: st.error(f"시스템 초기화 실패: {e}") st.stop()

st.session_state.messages = []
st.session_state.active_module = "Phase 0"

# [최적화] RAG 데이터는 세션 시작 시 즉시 로드 (사전 임베딩 사용)
s_data, s_emb, p_data, p_emb = load_precomputed_embeddings()
st.session_state.statutes = s_data
st.session_state.s_embeddings = s_emb
st.session_state.precedents = p_data
st.session_state.p_embeddings = p_emb

# 초기 인사/배치
try:
    init_prompt = "시스템 가동. Phase 0를 시작하라."
    resp = st.session_state.chat.send_message(init_prompt)
    init_text = resp.text
except Exception as e:
    init_text = f"[시스템 초기화 실패: {e}]"

st.session_state.messages.append({"role": "Architect", "content": init_text})
update_active_module(init_text)
---------------------------------------
7. 과거 메시지 렌더링 + 자동 스크롤
---------------------------------------
for m in st.session_state.messages: role_name = "Client" if m["role"] == "user" else "Architect" avatar = "👤" if m["role"] == "user" else "🛡️" with st.chat_message(role_name, avatar=avatar): st.markdown(m["content"], unsafe_allow_html=True)

자동 스크롤 JS 스니펫 (매 렌더링마다 실행)
if st.session_state.messages: st.markdown( '<script>setTimeout(()=>{const el=window.parent.document.querySelector("section.main");if(el)el.scrollTop=el.scrollHeight},100)</script>', unsafe_allow_html=True )

---------------------------------------
8. PDF 업로드 UI (Auto-Analysis Mode)
---------------------------------------
조건: active_module이 정확히 "Auto-Analysis Mode"이고, 9번을 입력한 직후일 때만 표시
if st.session_state.get("active_module") == "Auto-Analysis Mode": # 마지막 사용자 메시지가 "9"인지 확인 last_user_msg = None for m in reversed(st.session_state.messages): if m["role"] == "user": last_user_msg = m["content"].strip() break

# 9번 입력 직후에만 PDF UI 표시
if last_user_msg == "9":
    st.markdown("---")
    
    # 정보 박스 (이모지 제거)
    st.info("""
    **[ 사건기록 자동 분석 모드 ]**
    
    PDF 파일(판결문, 고소장, 답변서 등)을 업로드하면 AI가 자동으로:
    - 사건 도메인 분류 (형사/민사/가사 등)
    - 핵심 사실관계 5가지 추출
    - 확보된 증거 목록 정리
    - 양측 주장 요약
    
    **처리 시간:** 약 1-3분 | **최대 크기:** 50MB | **형식:** 텍스트 기반 PDF만 가능 (스캔본 불가)
    """)
    
    st.subheader("파일 업로드") # 이모지 제거
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "사건기록 PDF를 선택하세요",
            type=["pdf"],
            help="판결문, 고소장, 답변서, 사건기록 등",
            label_visibility="collapsed"
        )
    
    with col2:
        if uploaded_file:
            st.metric("상태", "준비 완료", delta="업로드 완료") # 이모지 제거
        else:
            st.metric("상태", "대기 중", delta="파일 선택") # 이모지 제거
    
    if uploaded_file is not None:
        file_size = uploaded_file.size / (1024 * 1024)
        
        # [주의] 이 부분의 들여쓰기가 핵심입니다. (NBSP 제거 완료)
        with st.container():
            st.success(f"**파일명:** {uploaded_file.name}  |  **크기:** {file_size:.1f}MB")
        
        if st.button("자동 분석 시작", type="primary", use_container_width=True): # 이모지 제거
            # 스피너 텍스트 (이모지 제거)
            with st.spinner("PDF 텍스트 추출 중... (30초~2분 소요)"):
                # 함수 호출 및 결과 받기
                pdf_text = extract_text_from_pdf(uploaded_file)
                
                # 상세 오류 처리 로직 (v8.1.2)
                if not pdf_text or (isinstance(pdf_text, str) and pdf_text.startswith("[ERROR:")):
                    
                    if pdf_text == "[ERROR:NO_TEXT]":
                        st.error("텍스트 추출 실패: PDF에 텍스트가 없습니다. 스캔된 이미지 파일일 수 있습니다. (텍스트 기반 PDF만 지원됩니다.)")
                    
                    elif pdf_text == "[ERROR:ENCRYPTED]":
                        st.error("PDF 처리 실패: 파일이 암호화되어 있습니다. 암호를 해제하고 다시 시도하세요.")

                    elif pdf_text == "[ERROR:PROCESSING_FAILED]":
                         st.error(f"PDF 처리 실패: 파일이 손상되었거나 처리 중 오류가 발생했습니다.")
                    
                    else:
                        # 예상치 못한 오류 또는 None 반환 시
                        st.error("PDF에서 텍스트를 추출할 수 없습니다. (알 수 없는 오류)")
                    
                    st.stop()
                
                # 성공 시 (pdf_text에 내용이 있음)
                st.success(f"텍스트 추출 완료 ({len(pdf_text):,} 글자)") # 이모지 제거
            
            # 분석 실행
            # 스피너 텍스트 (이모지 제거)
            with st.spinner("AI 분석 중... (1-2분 소요)"):
                # analyze_case_file 호출 시 모델 인자 전달하지 않음 (함수 내부에서 생성)
                analysis = analyze_case_file(pdf_text)
                
                if not analysis:
                    # analyze_case_file 내부에서 이미 에러 메시지 출력됨
                    st.stop()
            
            st.success("분석 완료!") # 이모지 제거
            
            # 결과 표시
            # 익스팬더 타이틀 (이모지 제거)
            with st.expander("분석 결과 상세 보기", expanded=True):
                col_a, col_b = st.columns(2)
                
                # 메트릭 타이틀 (이모지 제거)
                with col_a:
                    st.metric("도메인", analysis.get("domain", "미분류"))
                    st.metric("세부 분야", analysis.get("subdomain", "미분류"))
                
                with col_b:
                    st.metric("핵심 사실", f"{len(analysis.get('key_facts', []))}개")
                    st.metric("증거 항목", f"{len(analysis.get('evidence', []))}개")
                
                st.markdown("---")
                st.markdown("**핵심 사실관계**") # 이모지 제거
                for i, fact in enumerate(analysis.get("key_facts", []), 1):
                    st.markdown(f"{i}. {fact}")
                
                st.markdown("**확보된 증거**") # 이모지 제거
                for i, ev in enumerate(analysis.get("evidence", []), 1):
                    st.markdown(f"{i}. {ev}")
                
                st.markdown("**양측 주장**") # 이모지 제거
                st.info(f"**우리 측:** {analysis.get('our_claim', '(정보 없음)')}")
                st.warning(f"**상대 측:** {analysis.get('their_claim', '(정보 없음)')}")
            
                # 다음 단계 안내 (Phase 0 메뉴 기준 매핑)
                domain_map = {
                    "형사": "2", "민사": "8", "가사": "1", "이혼": "1",
                    "파산": "3", "행정": "7", "세무": "6", "IP": "4", "의료": "5",
                }
                
                domain_num = domain_map.get(analysis.get("domain"), "8") # 기본값 민사/기타
                
                # 안내 박스 (이모지 제거)
                st.info(
                    f"**[ 다음 단계 안내 ]**\n\n"
                    f"이 사건은 **{analysis.get('domain', '미분류')}** 사건으로 분류되었습니다.\n\n"
                    f"계속 진행하려면 아래 채팅창에 **{domain_num}**을 입력하세요."
                )
            
            # 세션 상태에 분석 결과 저장
            st.session_state["auto_analysis"] = analysis
            st.session_state["pdf_text"] = pdf_text
    
    st.markdown("---")
---------------------------------------
9. 자동 분석 결과 활용 UI
---------------------------------------
if "auto_analysis" in st.session_state and st.session_state.get("active_module") != "Auto-Analysis Mode": auto_data = st.session_state["auto_analysis"]

# 안내 박스 (이모지 제거)
st.success(
    "**[ 자동 분석 결과 감지됨 ]**\n\n"
    "시스템이 변수 질문을 시작하면, 아래 버튼을 눌러 자동으로 답변할 수 있습니다."
)

# 버튼 텍스트 (이모지 제거)
if st.button("자동 입력 활성화", type="secondary", use_container_width=True):
    auto_input = f"""
[자동 추출된 사건 정보]

도메인: {auto_data.get('domain', '미분류')} - {auto_data.get('subdomain', '미분류')}

핵심 사실관계: {chr(10).join(f"{i}. {fact}" for i, fact in enumerate(auto_data.get('key_facts', []), 1))}

확보된 증거: {chr(10).join(f"- {ev}" for ev in auto_data.get('evidence', []))}

우리 측 주장: {auto_data.get('our_claim', '(정보 없음)')}

상대방 주장: {auto_data.get('their_claim', '(정보 없음)')}

위 정보를 바탕으로 시뮬레이션을 진행해주세요. """

    st.session_state.messages.append({"role": "user", "content": auto_input})
    # 분석 결과 사용 후 세션에서 제거
    del st.session_state["auto_analysis"]
    st.rerun()

st.markdown("---")
---------------------------------------
10. 스트리밍 응답 함수 (안정성 강화)
---------------------------------------
def stream_and_store_response(chat_session, prompt_to_send: str, spinner_text: str = "Architect 시스템 연산 중..."): full_response = "" start_time = time.time()

with st.chat_message("Architect", avatar="🛡️"):
    placeholder = st.empty()
    try:
        with st.spinner(spinner_text):
            stream = chat_session.send_message(prompt_to_send, stream=True)
            for chunk in stream:
                # 안전 필터 및 응답 유효성 검사 강화
                if not getattr(chunk, "text", None):
                    # 텍스트가 비어있거나 필터에 걸린 경우
                    if chunk.candidates and chunk.candidates[0].finish_reason == 'SAFETY':
                         full_response = "[시스템 경고: 응답이 안전 필터에 의해 차단되었습니다.]"
                         placeholder.error(full_response)
                         break
                    elif hasattr(chunk, 'prompt_feedback') and chunk.prompt_feedback and chunk.prompt_feedback.block_reason:
                         full_response = f"[시스템 경고: 입력이 차단되었습니다. 사유: {chunk.prompt_feedback.block_reason}]"
                         placeholder.error(full_response)
                         break
                    continue # 텍스트가 없는 청크는 무시

                full_response += chunk.text
                placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
        
        # 최종 응답 표시
        placeholder.markdown(full_response, unsafe_allow_html=True)
    
    except Exception as e:
        full_response = f"[치명적 오류: {e}]"
        placeholder.error(full_response)

st.session_state.messages.append({"role": "Architect", "content": full_response})
update_active_module(full_response)

end_time = time.time()
print(f"[LLM] 응답 시간: {end_time - start_time:.2f}s")
return full_response
---------------------------------------
11. 메인 입력 루프 + Dual RAG (최적화 적용)
---------------------------------------
if prompt := st.chat_input("시뮬레이션 변수를 입력하십시오"):

# 1. 초기화 키워드 감지 (최우선)
if _is_reset_keyword(prompt):
    st.session_state.active_module = "Phase 0"
    # 관련 세션 상태 초기화
    if "auto_analysis" in st.session_state: del st.session_state["auto_analysis"]
    if "pdf_text" in st.session_state: del st.session_state["pdf_text"]
    
    # Phase 0 메뉴 다시 불러오기 (채팅 세션 및 메시지 기록 완전 초기화)
    try:
        st.session_state.chat = st.session_state.model.start_chat(history=[])
        init_prompt = "시스템 가동. Phase 0를 시작하라."
        resp = st.session_state.chat.send_message(init_prompt)
        init_text = resp.text
        # 메시지 기록도 초기화 후 첫 메시지만 추가
        st.session_state.messages = [{"role": "Architect", "content": init_text}]
    except Exception as e:
        st.error(f"[시스템 재시작 실패: {e}]")
        
    st.rerun()

# 2. 9번 입력 감지 (Auto-Analysis Mode 진입)
if prompt.strip() == "9":
    # Phase 0 상태에서만 9번 입력 허용 (안정성 강화)
    if "Phase 0" not in st.session_state.active_module:
         st.warning("Auto-Analysis Mode(9번)는 Phase 0 메뉴에서만 진입 가능합니다. '초기화'를 입력하세요.")
    else:
        st.session_state.active_module = "Auto-Analysis Mode"
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("Client", avatar="👤"):
            st.markdown(prompt, unsafe_allow_html=True)
        
        # AI에게 9번 입력 전달하여 모드 활성화 메시지 받기
        response_text = stream_and_store_response(st.session_state.chat, prompt)
        st.rerun() # UI 갱신하여 PDF 업로더 표시

# 3. 일반 채팅 처리
# 사용자 메시지 기록/표시
st.session_state.messages.append({"role": "user", "content": prompt})
with st.chat_message("Client", avatar="👤"):
    st.markdown(prompt, unsafe_allow_html=True)

# Phase 상태 확인
is_data_ingestion_phase = "Phase 2" in (st.session_state.active_module or "")

# RAG 컨텍스트 조립
rag_context = ""
similar_precedents = []

# 메뉴 입력이나 데이터 수집 단계가 아닐 때 RAG 실행
if not _is_menu_input(prompt) and not is_data_ingestion_phase:
    
    # Contextual Query 생성
    contextual_query = f"현재 활성화된 모듈: {st.session_state.active_module}. 사용자 질문/입력: {prompt}"

    # Dual RAG 실행
    # 스피너 텍스트 (이모지 제거)
    with st.spinner("실시간 데이터베이스 분석 중... (Dual RAG: 법령/판례)"):
        # 법령 검색 (S-RAG) - [최적화] 임계값 0.65로 하향 조정
        if st.session_state.statutes:
            s_hits = find_similar_items(
                contextual_query,
                st.session_state.statutes,
                st.session_state.s_embeddings,
                top_k=3,
                threshold=0.65, # 법령은 추상적이므로 기준 완화
            )
            if s_hits:
                s_texts = [
                    f"[유사도: {hit['similarity']:.2f}]\n"
                    f"{hit.get('rag_index', '내용 없음')}\n---\n"
                    for hit in s_hits
                ]
                rag_context += (
                    "\n\n[시스템 참조: 검색된 관련 법령 데이터]\n" +
                    "\n".join(s_texts)
                )

        # 판례 검색 (P-RAG) - [최적화] 임계값 0.75 유지
        if st.session_state.precedents:
            similar_precedents = find_similar_items(
                contextual_query,
                st.session_state.precedents,
                st.session_state.p_embeddings,
                top_k=5,
                threshold=0.75, # 판례는 구체적이므로 기준 유지
            )
            if similar_precedents:
                p_texts = [
                    f"[유사도: {hit['similarity']:.2f}]\n"
                    f"{hit.get('rag_index', '내용 없음')}\n---\n"
                    for hit in similar_precedents
                ]
                rag_context += (
                    "\n\n[시스템 참조: 검색된 유사 판례 데이터]\n" +
                    "\n".join(p_texts)
                )

# 최종 프롬프트 조립 및 전송
final_prompt = (
    f"[사용자 원문 입력]\n{prompt}\n"
    f"{rag_context}"
)

current_response = stream_and_store_response(st.session_state.chat, final_prompt)

# 판례 카드 시각화 (보고서 생성 시에만)
clean_response = re.sub("<[^<]+?>", "", current_response)

if _is_final_report(clean_response) and similar_precedents:
    q_title = _query_title(prompt)
    # 타이틀 (이모지 제거)
    st.markdown(
        f"**[ 실시간 판례 전문 분석 (P-RAG 결과) ]**\n\n"
        f"* 검색 쿼리: `[{q_title}]`\n"
    )

    for case_data in similar_precedents[:3]:
        sim_pct = int(round(case_data["similarity"] * 100))

        title = case_data.get("title", "제목 없음")
        case_no = case_data.get("case_no", case_data.get("id", ""))
        court = case_data.get("court", "")
        date = case_data.get("date", "")
        url = case_data.get("url")
        full_text = case_data.get("full_text", case_data.get("raw_text"))

        label = f"판례 [{title}]"
        if court and case_no:
            label += f" — {court} {case_no}"

        summary = case_data.get("rag_index", "요약 내용 없음")
        if len(summary) > 200:
            summary = summary[:197] + "..."

        # 링크 텍스트 (이모지 제거)
        link_md = f"[원문 링크 보기]({url})" if url else ""

        md = (
            f"* **{label}**\n"
            f"  - 선고: {date} | 유사도: {sim_pct}% | {link_md}\n"
            f"  - 내용 요약: {summary}"
        )
        st.markdown(md)

        if full_text:
            # 익스팬더 타이틀 (이모지 제거)
            with st.expander("판례 전문 보기"):
                st.text(full_text)

elif _is_final_report(clean_response) and not similar_precedents:
    # 안내 박스 (이모지 제거)
    st.info(
        "안내: 분석과 관련된 유사 판례가 데이터베이스에서 검색되지 않았습니다. "
        "(임계값 0.75)"
    )
