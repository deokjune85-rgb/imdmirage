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

# API 키 설정
if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    # 로컬 테스트용 (secrets.toml 없을 경우)
    st.warning("Google API Key가 설정되지 않았습니다.")

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
    if not clean_text: return None
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
    if not items or not embeddings: return []
    try:
        q_emb = embed_text(query_text, task_type="retrieval_query")
        if q_emb is None: return []
        sims = np.dot(np.array(embeddings), np.array(q_emb))
        idxs = np.argsort(sims)[::-1][:top_k]
        results = []
        for i in idxs:
            score = float(sims[i])
            if score < threshold: continue
            item = items[i].copy()
            item["similarity"] = score
            results.append(item)
        return results
    except Exception as e:
        print(f"[RAG Error] {e}")
        return []

# ---------------------------------------
# 2. 데이터 로드 함수 (파일 기반 RAG)
# ---------------------------------------
@st.cache_resource
def load_precomputed_embeddings():
    statute_items = []
    statute_embeddings = []
    precedent_items = []
    precedent_embeddings = []

    # 로딩 상태 시각화
    with st.status("📚 Veritas 데이터베이스 연결 중...", expanded=True) as status:
        try:
            # 1. 법령 로드
            if os.path.exists("statutes_data.txt"):
                st.write("📜 법령 데이터 스캔...")
                with open("statutes_data.txt", "r", encoding="utf-8") as f:
                    content = f.read()
                parts = re.split(r"\s*---END OF STATUTE---\s*", content)
                for i, p in enumerate(parts):
                    if i >= 10: break # 데모용 제한
                    p = p.strip()
                    if not p: continue
                    emb = embed_text(p)
                    if emb:
                        statute_items.append({"rag_index": p, "raw_text": p})
                        statute_embeddings.append(emb)
                        time.sleep(0.1)
                st.write(f"✅ 법령 {len(statute_items)}건 로드")
            
            # 2. 판례 로드
            if os.path.exists("precedents_data.jsonl"):
                st.write("⚖️ 판례 데이터 스캔...")
                with open("precedents_data.jsonl", "r", encoding="utf-8") as f:
                    count = 0
                    for line in f:
                        if count >= 10: break # 데모용 제한
                        line = line.strip()
                        if not line: continue
                        try:
                            obj = json.loads(line)
                            txt = obj.get("rag_index", "")
                            if txt:
                                emb = embed_text(txt)
                                if emb:
                                    precedent_items.append(obj)
                                    precedent_embeddings.append(emb)
                                    count += 1
                                    time.sleep(0.1)
                        except: continue
                st.write(f"✅ 판례 {len(precedent_items)}건 로드")
            
            status.update(label="시스템 준비 완료", state="complete", expanded=False)
        except Exception as e:
            print(f"[RAG 로딩 에러] {e}")

    return statute_items, statute_embeddings, precedent_items, precedent_embeddings

# ---------------------------------------
# 3. PDF 처리 함수
# ---------------------------------------
def extract_text_from_pdf(uploaded_file):
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"PDF 처리 오류: {e}")
        return None

def analyze_case_file(pdf_text: str, model):
    analysis_prompt = f"""
    다음 사건 기록을 정밀 분석하여 JSON 형식으로 출력하십시오.
    {{
        "domain": "형사/민사/가사",
        "subdomain": "세부 죄명 또는 쟁점",
        "key_facts": ["사실1", "사실2", "사실3", "사실4", "사실5"],
        "evidence": ["증거1", "증거2"],
        "our_claim": "우리 측 주장",
        "their_claim": "상대방 주장"
    }}
    [내용]
    {pdf_text[:10000]}
    """
    try:
        response = model.generate_content(analysis_prompt, generation_config={"response_mime_type": "application/json"})
        return json.loads(response.text)
    except:
        return None

# ---------------------------------------
# 4. 유틸 및 스트리밍 함수
# ---------------------------------------
def _is_reset_keyword(s: str) -> bool:
    return any(kw in s.lower() for kw in ["처음", "메인", "초기화", "reset"])

def update_active_module(response_text: str):
    if ("9." in response_text and "사건기록" in response_text) or \
       ("Auto-Analysis Mode" in response_text):
        st.session_state.active_module = "Auto-Analysis Mode"
    
    m = re.search(r"'(.+?)' 모듈을 (?:최종 )?활성화합니다", response_text)
    if m:
        st.session_state.active_module = m.group(1).strip()

def stream_and_store_response(chat_session, prompt_to_send: str):
    full_response = ""
    with st.chat_message("Architect", avatar="🛡️"):
        placeholder = st.empty()
        try:
            stream = chat_session.send_message(prompt_to_send, stream=True)
            for chunk in stream:
                if getattr(chunk, "text", None):
                    full_response += chunk.text
                    placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
        except Exception as e:
            placeholder.error(f"연산 오류: {e}")
    
    st.session_state.messages.append({"role": "Architect", "content": full_response})
    update_active_module(full_response)
    return full_response

# ---------------------------------------
# 5. 메인 로직
# ---------------------------------------

# 모델 초기화
if "model" not in st.session_state:
    try:
        # Gemini 1.5 Flash 사용
        st.session_state.model = genai.GenerativeModel("models/gemini-1.5-flash", system_instruction=SYSTEM_INSTRUCTION)
        st.session_state.chat = st.session_state.model.start_chat(history=[])
        st.session_state.messages = []
        
        # [핵심 수정] 초기 메시지 강제 주입 (화면에 글 안 나오는 문제 해결)
        init_msg = """
        **Veritas Engine 8.1 가동.**
        
        법률 전략 수립을 위한 Architect가 준비되었습니다.
        원하시는 작업이나 사건 개요를 입력하십시오.
        
        (PDF 분석을 원하시면 '9'를 입력하십시오)
        """
        st.session_state.messages.append({"role": "Architect", "content": init_msg})
        st.session_state.active_module = "Phase 0"
        
        # 데이터 로드
        s_data, s_emb, p_data, p_emb = load_precomputed_embeddings()
        st.session_state.statutes = s_data
        st.session_state.s_embeddings = s_emb
        st.session_state.precedents = p_data
        st.session_state.p_embeddings = p_emb
        
    except Exception as e:
        st.error(f"초기화 실패: {e}")

# 대화 내역 출력
for m in st.session_state.messages:
    avatar = "🛡️" if m["role"] == "Architect" else "👤"
    with st.chat_message(m["role"], avatar=avatar):
        st.markdown(m["content"])

# 화면 스크롤 하단 고정
st.markdown('<script>window.scrollTo(0, document.body.scrollHeight);</script>', unsafe_allow_html=True)

# PDF 모드 UI
if st.session_state.get("active_module") == "Auto-Analysis Mode":
    # 마지막 대화가 '9'일 때만 표시
    if st.session_state.messages and st.session_state.messages[-1]["content"] == "9":
        st.markdown("---")
        st.info("📄 **사건기록 PDF 자동 분석 모드**")
        uploaded_file = st.file_uploader("판결문/고소장 PDF 업로드", type=["pdf"])
        
        if uploaded_file:
            if st.button("🚀 자동 분석 시작", type="primary", use_container_width=True):
                with st.spinner("텍스트 추출 및 AI 분석 중..."):
                    pdf_text = extract_text_from_pdf(uploaded_file)
                    if pdf_text:
                        analysis = analyze_case_file(pdf_text, st.session_state.model)
                        if analysis:
                            st.success("분석 완료")
                            with st.expander("📊 분석 결과 보기", expanded=True):
                                st.markdown(f"**도메인:** {analysis.get('domain')}")
                                st.markdown("**핵심 사실:**")
                                for f in analysis.get('key_facts', []):
                                    st.markdown(f"- {f}")
                            
                            # 자동 진행
                            st.session_state["auto_analysis"] = analysis
                            
                            # 다음 단계 자동 트리거
                            domain_map = {"형사": "2", "민사": "8", "이혼": "1"}
                            domain_num = domain_map.get(analysis.get("domain", ""), "8")
                            
                            auto_prompt = f"""
                            [자동 분석 데이터]
                            도메인: {analysis.get('domain')}
                            사실관계: {analysis.get('key_facts')}
                            양측주장: {analysis.get('our_claim')} vs {analysis.get('their_claim')}
                            
                            위 데이터를 바탕으로 {domain_num}번 모듈을 실행하여 승소 전략을 제시하라.
                            """
                            
                            st.session_state.messages.append({"role": "user", "content": "PDF 분석 완료. 자동 전략 수립 시작."})
                            stream_and_store_response(st.session_state.chat, auto_prompt)
                            st.rerun()

# 채팅 입력
if prompt := st.chat_input("명령 또는 사건 내용을 입력하십시오..."):
    # 초기화
    if _is_reset_keyword(prompt):
        st.session_state.chat = st.session_state.model.start_chat(history=[])
        st.session_state.messages = [{"role": "Architect", "content": "시스템이 리셋되었습니다. 처음부터 다시 시작합니다."}]
        st.rerun()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("Client", avatar="👤"):
        st.markdown(prompt)
    
    if prompt.strip() == "9":
        st.session_state.active_module = "Auto-Analysis Mode"
        stream_and_store_response(st.session_state.chat, "사건기록 자동 분석 모드에 대해 설명하라.")
        st.rerun()
    else:
        # RAG 검색 (법률/판례가 있다면 컨텍스트 추가)
        rag_context = ""
        if st.session_state.statutes:
            sim_statutes = find_similar_items(prompt, st.session_state.statutes, st.session_state.s_embeddings)
            if sim_statutes:
                rag_context += "\n[관련 법령]\n" + "\n".join([s['raw_text'] for s in sim_statutes])
        
        if rag_context:
            full_prompt = f"사용자 질문: {prompt}\n\n{rag_context}\n\n위 법적 근거를 참고하여 답변하라."
        else:
            full_prompt = prompt
            
        stream_and_store_response(st.session_state.chat, full_prompt)
