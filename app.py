import streamlit as st
from dotenv import load_dotenv
from utils import *
import uuid

if 'unique_id' not in st.session_state:
    st.session_state['unique_id'] = ''

def main():
    load_dotenv()

    st.set_page_config(page_title="Resume Screening Assistance")
    st.title("HR - Assistant by Bitan⚡")
    st.subheader("I can help you in resume screening process")

    job_description = st.text_area("Please paste the 'JOB DESCRIPTION' here...", key="1")
    document_count = st.text_input("No.of 'RESUMES' to return", key="2")
    pdf = st.file_uploader("Upload resumes here, only PDF files allowed", type=["pdf"], accept_multiple_files=True)

    submit = st.button("Help me with the analysis")

    if submit:
        with st.spinner('Wait for it...'):
            st.session_state['unique_id'] = uuid.uuid4().hex
            
            final_docs_list = create_docs(pdf, st.session_state['unique_id'])
            st.write("*Resumes uploaded* :" + str(len(final_docs_list)))

            embeddings = create_embeddings_load_data()
            
            vector_store = create_vector_store(embeddings, final_docs_list)

            relevant_docs = similar_docs(job_description, int(document_count), vector_store, st.session_state['unique_id'])

            st.write(":heavy_minus_sign:" * 30)

            for item, (doc, score) in enumerate(relevant_docs, 1):
                st.subheader(f"👉 {item}")
                st.write(f"**File** : {doc.metadata['name']}")
                
                with st.expander('Show me 👀'): 
                    st.info(f"**Match Score** : {score:.4f}")
                    summary = get_summary(doc)
                    st.write(f"**Summary** : {summary}")

        st.success("Hope I was able to save your time❤️")

if __name__ == '__main__':
    main()
