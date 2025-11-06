import streamlit as st
import google.generativeai as genai
import os

# --- 1. 시스템 설정 (The Vault & Mirage Protocol) ---
# 페이지 타이틀과 레이아웃 설정
st.set_page_config(page_title="아이엠디 아키텍처 버전 7.0", page_icon="🛡️", layout="centered")

# CSS 해킹: Streamlit 기본 로고, 메뉴, 헤더, 푸터를 완벽하게 숨긴다. (신기루 프로토콜)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .stDeployButton {visibility: hidden;} /* 배포 버튼 숨기기 */
            /* 필요시 여기에 추가적인 커스텀 스타일링(예: 다크 모드)을 적용하라. */
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- 2. 타이틀 및 경고 (황제의 교리) ---
st.title("아이엠디 아키텍처 버전 7.0")
# st.error를 사용하여 강력한 시각적 경고 표시
st.error("보안 경고: 본 시스템은 격리된 사설 환경(The Vault)에서 작동합니다. 모든 데이터는 기밀로 취급되며 외부로 유출되지 않습니다.")

# --- 3. API 키 및 모델 설정 ---
try:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("시스템 오류: 엔진 연결 실패. (API Key 누락)")
    st.stop()

genai.configure(api_key=API_KEY)

# ← 외부 파일 로드 (유일한 선언!)
with open("system_prompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_INSTRUCTION = f.read()

# ← models/ 강제 + 1.5-flash-latest
st.session_state.model = genai.GenerativeModel(
    'models/gemini-1.5-flash-latest',
    system_instruction=SYSTEM_INSTRUCTION
)
