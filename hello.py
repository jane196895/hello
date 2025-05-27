import streamlit as st
import tempfile
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# 문서 로딩 및 문단 추출
def extract_paragraphs_with_meta(uploaded_files):
    paragraphs = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        doc = fitz.open(tmp_path)
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            chunks = [chunk.strip() for chunk in text.split("\n\n") if len(chunk.strip()) > 50]
            for chunk in chunks:
                paragraphs.append({
                    "text": chunk,
                    "source": file.name,
                    "page": page_num
                })
        doc.close()
    return paragraphs

# 유사한 문단 상위 3개 찾기
def find_top_k_matches(question, paragraphs, k=3):
    texts = [p["text"] for p in paragraphs]
    vectorizer = TfidfVectorizer().fit(texts + [question])
    tfidf_matrix = vectorizer.transform(texts + [question])
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    top_k_indices = similarities.argsort()[::-1][:k]
    results = []
    for idx in top_k_indices:
        match = paragraphs[idx].copy()
        match["similarity"] = similarities[idx]
        results.append(match)
    return results

# Streamlit UI
st.set_page_config(page_title="학생생활지도 문서 FAQ 챗봇", layout="centered")
st.title("📄 문서 기반 학생생활지도 FAQ 챗봇 (API 없이 작동)")

uploaded_files = st.file_uploader("📂 교육청 PDF 문서 업로드", type="pdf", accept_multiple_files=True)
situation_input = st.text_area("📝 학생 상황을 입력하세요:", placeholder="예: 학생이 친구 물건을 몰래 가져갔어요.")

if st.button("📌 관련 지침 확인하기"):
    if not situation_input.strip():
        st.warning("학생 상황을 입력해주세요.")
    elif not uploaded_files:
        st.warning("문서를 업로드해주세요.")
    else:
        with st.spinner("문서 분석 중입니다..."):
            paragraphs = extract_paragraphs_with_meta(uploaded_files)
            top_matches = find_top_k_matches(situation_input, paragraphs)

        st.markdown("## 🧭 관련 지침 및 문서 내용 (Top 3 유사 문단)")

        for i, match in enumerate(top_matches, start=1):
            st.markdown(f"""
### {i}. 📘 문서: *{match['source']}* (📄 {match['page']}쪽)
- **유사도 점수**: `{match['similarity']:.2f}`

> {match['text']}
---
""")

        st.info("위 문단들은 업로드한 문서에서 질문과 가장 유사한 내용을 기반으로 추출된 것입니다.")

