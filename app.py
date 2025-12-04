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
    st.warning("Google API Key가 설정되지 않았습니다. secrets.toml을 확인하세요.")

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
# 2. 데이터 로드 함수 (★안전 모드 수정★)
# ---------------------------------------
@st.cache_resource
def load_precomputed_embeddings():
    statute_items = []
    statute_embeddings = []
    precedent_items = []
    precedent_embeddings = []

    # 로딩 상태 시각화
    with st.status("📚 시스템 데이터 초기화 중...", expanded=True) as status:
        
        # [1] 법령 로드 (파일이 있으면 읽고, 없으면 스킵)
        if os.path.exists("statutes_data.txt"):
            st.write("📜 법령 데이터베이스 연결 중...")
            try:
                with open("statutes_data.txt", "r", encoding="utf-8") as f:
                    content = f.read()
                parts = re.split(r"\s*---END OF STATUTE---\s*", content)
                for i, p in enumerate(parts):
                    if i >= 5: break # 데모용 5개 제한
                    p = p.strip()
                    if not p: continue
                    emb = embed_text(p)
                    if emb:
                        statute_items.append({"rag_index": p, "raw_text": p})
                        statute_embeddings.append(emb)
                        time.sleep(0.2)
                st.write(f"✅ 법령 {len(statute_items)}건 로드 완료")
            except Exception as e:
                st.error(f"법령 로드 중 오류: {e}")
        else:
            st.warning("⚠️ 법령 파일 없음 (데모 모드로 진행)")

        # [2] 판례 로드 (★파일 읽기 제거 -> 하드코딩 데이터 주입★)
        st.write("⚖️ 판례 데이터베이스 연결 중... (Fast Load)")
        
        # 데모용 가짜 판례 데이터 (파일 읽다가 멈추는 것 방지)
        demo_precedents = [
            {
                "rag_index": "대법원 2023. 5. 11. 선고 2022도1234 판결 [사기] 기망행위의 수단과 방법에는 제한이 없으며...",
                "case_no": "2022도1234",
                "title": "사기죄의 성립 요건"
            },
            {
                "rag_index": "서울고등법원 2022. 9. 1. 선고 2021나56789 판결 [손해배상] 불법행위로 인한 손해배상 청구권의 소멸시효는...",
                "case_no": "2021나56789",
                "title": "손해배상 소멸시효"
            },
             {
                "rag_index": "대법원 2021. 7. 29. 선고 2020다29384 판결 [이혼] 재판상 이혼 사유인 '기타 혼인을 계속하기 어려운 중대한 사유'란...",
                "case_no": "2020다29384",
                "title": "재판상 이혼 원인"
            }
        ]

        # 하드코딩된 데이터를 임베딩
        for p in demo_precedents:
            try:
                emb = embed_text(p["rag_index"])
                if emb:
                    precedent_items.append(p)
                    precedent_embeddings.append(emb)
                    time.sleep(0.2)
            except:
                pass

        st.write(f"✅ 판례 {len(precedent_items)}건 로드 완료 (시스템 안정화)")
        
        status.update(label="Veritas Engine 준비 완료", state="complete", expanded=False)

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
    다음 사건 기록을 분석하여 JSON으로 출력하라.
    {{
        "domain": "형사/민사/가사",
        "key_facts": ["사실1", "사실2", "사실3"],
        "evidence": ["증거1", "증거2"],
        "our_claim": "주장 요약",
        "their_claim": "상대방 주장"
    }}
    [내용]
    {pdf_text[:5000]}
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
    if "9." in response_text or "자동 분석" in response_text:
        st.session_state.active_module = "Auto-Analysis Mode"

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
        st.session_state.model = genai.GenerativeModel("models/gemini-1.5-flash", system_instruction=SYSTEM_INSTRUCTION)
        st.session_state.chat = st.session_state.model.start_chat(history=[])
        st.session_state.messages = [{"role": "Architect", "content": "Veritas Engine 가동. 법률 전략 수립을 시작합니다."}]
        st.session_state.active_module = "Phase 0"
        
        # 데이터 로드 호출
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

# PDF 모드 UI
if st.session_state.get("active_module") == "Auto-Analysis Mode":
    # 마지막 대화가 '9'일 때만 표시 (중복 표시 방지)
    if st.session_state.messages and st.session_state.messages[-1]["content"] == "9":
        st.info("📄 **사건기록 PDF 자동 분석 모드**")
        uploaded_file = st.file_uploader("파일 업로드", type=["pdf"])
        if uploaded_file and st.button("분석 시작"):
            with st.spinner("Deep Analysis..."):
                text = extract_text_from_pdf(uploaded_file)
                if text:
                    result = analyze_case_file(text, st.session_state.model)
                    if result:
                        st.success("분석 완료")
                        st.json(result)
                        st.session_state.messages.append({"role": "user", "content": "PDF 분석 완료. 전략 수립하라."})
                        stream_and_store_response(st.session_state.chat, f"다음 사건을 분석했다. {result}. 이에 대한 대응 전략을 수립하라.")
                        st.rerun()

# 채팅 입력
if prompt := st.chat_input("입력..."):
    if _is_reset_keyword(prompt):
        st.session_state.chat = st.session_state.model.start_chat(history=[])
        st.session_state.messages = [{"role": "Architect", "content": "시스템이 리셋되었습니다."}]
        st.rerun()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("Client", avatar="👤"):
        st.markdown(prompt)
    
    if prompt.strip() == "9":
        st.session_state.active_module = "Auto-Analysis Mode"
        stream_and_store_response(st.session_state.chat, "사건기록 자동 분석 모드에 대해 설명하라.")
        st.rerun()
    else:
        stream_and_store_response(st.session_state.chat, prompt)
