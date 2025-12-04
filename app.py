import streamlit as st
import google.generativeai as genai
import os
import re
import json
import numpy as np
import PyPDF2
import time

# ---------------------------------------
# 0. 시스템 설정
# ---------------------------------------
st.set_page_config(
    page_title="Veritas Engine 8.1 | Legal Architect",
    page_icon="⚖️",
    layout="centered"
)

# API 키 설정 (Streamlit Secrets에서 가져오거나 환경변수 사용)
# st.secrets["GOOGLE_API_KEY"] 설정이 필요합니다.
if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    st.error("Google API Key가 설정되지 않았습니다.")
    st.stop()

SYSTEM_INSTRUCTION = """
당신은 대한민국 최고의 법률 전문가이자 전략가인 'Veritas Architect'입니다.
사용자의 질문이나 사건 기록을 분석하여 법리적 근거(조문, 판례)에 기반한 명확한 전략을 제시하십시오.
"""

EMBEDDING_MODEL_NAME = "models/text-embedding-004"

# ---------------------------------------
# 1. 임베딩 및 RAG 검색 함수
# ---------------------------------------
def embed_text(text: str, task_type: str = "retrieval_document"):
    """텍스트 임베딩 생성"""
    clean_text = text.replace("\n", " ").strip()
    if not clean_text:
        return None
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL_NAME,
            content=clean_text,
            task_type=task_type
        )
        return result['embedding']
    except Exception as e:
        print(f"[Embedding error] {e}")
        return None

def find_similar_items(query_text, items, embeddings, top_k=3, threshold=0.5):
    """유사도 검색"""
    if not items or not embeddings:
        return []

    try:
        q_emb = embed_text(query_text, task_type="retrieval_query")
        if q_emb is None:
            return []

        sims = np.dot(np.array(embeddings), np.array(q_emb))
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
    except Exception as e:
        print(f"[RAG Error] {e}")
        return []

# ---------------------------------------
# 2. 데이터 로드 함수 (사전 임베딩)
# ---------------------------------------
@st.cache_resource
def load_precomputed_embeddings():
    statute_items = []
    statute_embeddings = []
    precedent_items = []
    precedent_embeddings = []

    try:
        # 법령 로드
        if os.path.exists("statutes_data.txt"):
            with open("statutes_data.txt", "r", encoding="utf-8") as f:
                content = f.read()
            
            parts = re.split(r"\s*---END OF STATUTE---\s*", content)
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                # 실제로는 임베딩 값이 저장된 파일을 로드하거나, 여기서 생성 (시간 소요됨)
                # 여기서는 데모용으로 실시간 생성 로직 유지
                emb = embed_text(p)
                if emb:
                    statute_items.append({"rag_index": p, "raw_text": p})
                    statute_embeddings.append(emb)
            
            print(f"[RAG] ✅ 법령 로드: {len(statute_items)}개")
        
        # 판례 로드
        if os.path.exists("precedents_data.jsonl"):
            with open("precedents_data.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        txt = obj.get("rag_index", "")
                        if txt:
                            emb = embed_text(txt)
                            if emb:
                                precedent_items.append(obj)
                                precedent_embeddings.append(emb)
                    except:
                        continue
            
            print(f"[RAG] ✅ 판례 로드: {len(precedent_items)}개")
            
    except Exception as e:
        print(f"[RAG 로딩 에러] {e}")

    return statute_items, statute_embeddings, precedent_items, precedent_embeddings

# ---------------------------------------
# 3. PDF 처리 함수
# ---------------------------------------
def extract_text_from_pdf(uploaded_file):
    """PDF에서 텍스트 추출"""
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"PDF 읽기 오류: {e}")
        return None

def analyze_case_file(pdf_text: str, model):
    """PDF 내용 자동 분석"""
    analysis_prompt = f"""
    다음은 사건기록 PDF에서 추출한 내용입니다. 
    내용을 정밀 분석하여 다음 JSON 형식으로 출력하십시오.
    
    {{
        "domain": "형사/민사/가사 중 택1",
        "subdomain": "세부 죄명 또는 쟁점 (예: 사기, 손해배상)",
        "key_facts": ["핵심 사실관계1", "핵심 사실관계2", ... (5개 내외)],
        "evidence": ["확보된 증거1", "확보된 증거2", ...],
        "our_claim": "우리 측 핵심 주장 요약",
        "their_claim": "상대방 핵심 주장 요약"
    }}

    [사건 내용]
    {pdf_text[:10000]}
    """
    try:
        response = model.generate_content(analysis_prompt, generation_config={"response_mime_type": "application/json"})
        return json.loads(response.text)
    except Exception as e:
        st.error(f"AI 분석 오류: {e}")
        return None

# ---------------------------------------
# 4. 유틸리티 함수
# ---------------------------------------
def _is_reset_keyword(s: str) -> bool:
    """초기화 키워드 감지"""
    keywords = ["처음", "메인", "초기화", "reset", "돌아가", "처음으로"]
    return any(kw in s.lower() for kw in keywords)

def _is_final_report(txt: str) -> bool:
    return "전략 브리핑 보고서" in txt

def update_active_module(response_text: str):
    """활성 모듈 상태 업데이트"""
    if ("9." in response_text and "사건기록 자동 분석 모드" in response_text) or \
       ("Auto-Analysis Mode를 활성화합니다" in response_text):
        st.session_state.active_module = "Auto-Analysis Mode"
        return

    m = re.search(r"'(.+?)' 모듈을 (?:최종 )?활성화합니다", response_text)
    if m:
        st.session_state.active_module = m.group(1).strip()

def stream_and_store_response(chat_session, prompt_to_send: str):
    """스트리밍 응답 처리 및 저장"""
    full_response = ""
    
    with st.chat_message("Architect", avatar="🛡️"):
        placeholder = st.empty()
        try:
            with st.spinner("Architect 시스템 연산 중..."):
                stream = chat_session.send_message(prompt_to_send, stream=True)
                for chunk in stream:
                    if not getattr(chunk, "parts", None):
                        full_response = "[시스템 경고: 응답이 안전 필터에 의해 차단되었습니다.]"
                        placeholder.error(full_response)
                        break
                    text_chunk = chunk.text
                    full_response += text_chunk
                    placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
            
            placeholder.markdown(full_response, unsafe_allow_html=True)
            
        except Exception as e:
            full_response = f"[치명적 오류 발생: {e}]"
            placeholder.error(full_response)

    st.session_state.messages.append({"role": "Architect", "content": full_response})
    update_active_module(full_response)
    return full_response

# ---------------------------------------
# 5. 메인 앱 로직
# ---------------------------------------

# 세션 초기화
if "model" not in st.session_state:
    try:
        st.session_state.model = genai.GenerativeModel(
            "models/gemini-2.0-flash-exp", # 최신 모델 사용
            system_instruction=SYSTEM_INSTRUCTION,
        )
        st.session_state.chat = st.session_state.model.start_chat(history=[])
        st.session_state.messages = []
        st.session_state.active_module = "Phase 0"
        
        # 초기 인사말
        init_msg = "Veritas Engine 8.1 가동. 법률 전략 수립을 시작합니다."
        st.session_state.messages.append({"role": "Architect", "content": init_msg})
        
        # RAG 데이터 로드
        s_data, s_emb, p_data, p_emb = load_precomputed_embeddings()
        st.session_state.statutes = s_data
        st.session_state.s_embeddings = s_emb
        st.session_state.precedents = p_data
        st.session_state.p_embeddings = p_emb
        
    except Exception as e:
        st.error(f"시스템 초기화 실패: {e}")

# 채팅 히스토리 출력
for m in st.session_state.messages:
    role = m["role"]
    avatar = "🛡️" if role == "Architect" else "👤"
    with st.chat_message(role, avatar=avatar):
        st.markdown(m["content"], unsafe_allow_html=True)

# 화면 스크롤 하단 고정
st.markdown('<script>window.scrollTo(0, document.body.scrollHeight);</script>', unsafe_allow_html=True)


# ---------------------------------------
# 6. PDF 업로드 UI (Auto-Analysis Mode)
# ---------------------------------------
if st.session_state.get("active_module") == "Auto-Analysis Mode":
    # 마지막 메시지가 9번 선택인 경우에만 표시
    last_user_msg = ""
    for m in reversed(st.session_state.messages):
        if m["role"] == "user":
            last_user_msg = m["content"].strip()
            break
            
    if last_user_msg == "9":
        st.markdown("---")
        st.info("""
        **📄 사건기록 자동 분석 모드**
        판결문, 고소장 등 PDF 파일을 업로드하면 AI가 자동으로 쟁점을 추출하고 전략을 수립합니다.
        """)
        
        uploaded_file = st.file_uploader("사건기록 PDF 선택", type=["pdf"])
        
        if uploaded_file:
            if st.button("🚀 자동 분석 시작", type="primary"):
                with st.spinner("텍스트 추출 및 분석 중..."):
                    pdf_text = extract_text_from_pdf(uploaded_file)
                    if pdf_text:
                        analysis = analyze_case_file(pdf_text, st.session_state.model)
                        if analysis:
                            # 분석 결과 표시
                            with st.expander("📊 분석 결과 요약", expanded=True):
                                st.markdown(f"**도메인:** {analysis.get('domain')}")
                                st.markdown("**핵심 사실:**")
                                for f in analysis.get('key_facts', []):
                                    st.markdown(f"- {f}")
                            
                            # 자동 진행 로직
                            st.session_state["auto_analysis"] = analysis
                            
                            # 다음 단계 자동 트리거 메시지 생성
                            domain_map = {"형사": "2", "민사": "8", "이혼": "1"}
                            domain_num = domain_map.get(analysis.get("domain", ""), "8")
                            
                            auto_prompt = f"""
                            [자동 분석 데이터]
                            도메인: {analysis.get('domain')}
                            사실관계: {analysis.get('key_facts')}
                            
                            위 데이터를 바탕으로 {domain_num}번 모듈을 실행하여 전략을 제시하라.
                            """
                            
                            # 챗봇에게 자동 전송 효과
                            st.session_state.messages.append({"role": "user", "content": "PDF 분석 완료. 자동 전략 수립 시작."})
                            stream_and_store_response(st.session_state.chat, auto_prompt)
                            st.rerun()

# ---------------------------------------
# 7. 사용자 입력 처리
# ---------------------------------------
if prompt := st.chat_input("명령 또는 내용을 입력하십시오."):
    # 1. 초기화 감지
    if _is_reset_keyword(prompt):
        st.session_state.active_module = "Phase 0"
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.chat = st.session_state.model.start_chat(history=[]) # 대화 내역 초기화
        
        reset_msg = "시스템을 초기화합니다. 메인 메뉴로 돌아갑니다."
        st.session_state.messages.append({"role": "Architect", "content": reset_msg})
        
        # 메인 메뉴 호출
        stream_and_store_response(st.session_state.chat, "시스템 메뉴를 출력하라.")
        st.rerun()

    # 2. 일반 대화
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("Client", avatar="👤"):
        st.markdown(prompt)

    # 3. 9번(PDF 모드) 진입 감지
    if prompt.strip() == "9":
        st.session_state.active_module = "Auto-Analysis Mode"
        response_text = stream_and_store_response(st.session_state.chat, prompt)
        st.rerun()
    else:
        # 일반 응답 생성
        stream_and_store_response(st.session_state.chat, prompt)
