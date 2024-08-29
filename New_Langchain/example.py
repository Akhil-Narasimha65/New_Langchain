import os
import time
from dotenv import load_dotenv
from pdfminer.high_level import extract_text
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
import re
from openai import OpenAIError

# Load environment variables
load_dotenv()

# Set your OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Function to handle embedding with improved retry logic
def embed_with_retry(embedding_func, *args, **kwargs):
    max_retries = 10
    base_delay = 2
    max_delay = 60

    for attempt in range(max_retries):
        try:
            return embedding_func(*args, **kwargs)
        except OpenAIError as e:
            if "RateLimitError" in str(e):
                if attempt < max_retries - 1:
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    st.warning(f"Rate limit reached. Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                else:
                    st.error("Max retries reached. Please try again later or consider upgrading your OpenAI plan.")
                    return None
            else:
                st.error(f"An OpenAI error occurred: {str(e)}")
                return None

# Function to process the PDF using PyMuPDF
def process_pdf(file):
    with open("temp.pdf", "wb") as f:
        f.write(file.getbuffer())
    
    text = extract_text("temp.pdf")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = embed_with_retry(FAISS.from_texts, chunks, embedding=embeddings)
    return db

def extract_numbers_with_context(text):
    pattern = r'([A-Za-z\s]+)\s*(\d+(?:\.\d+)?)\s*([A-Za-z\s]+)'
    matches = re.findall(pattern, text)
    return matches



def post_process_aggregation(question, answer):
    """Post-process the answer for aggregation questions with improved filtering and accuracy."""
    lower_question = question.lower()
    
    # Extract numbers and context
    numbers_with_context = extract_numbers_with_context(answer)
    
    # Determine if the question is asking for an aggregation
    aggregation_keywords = ["total", "sum", "aggregate", "overall", "count", "total sales"]
    is_aggregation_question = any(keyword in lower_question for keyword in aggregation_keywords)
    
    # Extract the filter condition from the question
    filter_words = ["region", "person", "division", "territory", "representative"]
    filter_condition = next((word for word in filter_words if word in lower_question), None)
    
    if is_aggregation_question:
        if filter_condition:
            filter_value_match = re.search(fr"{filter_condition}\s+(\w+)", lower_question, re.IGNORECASE)
            if filter_value_match:
                filter_value = filter_value_match.group(1).lower()
                filtered_numbers = []
                
                for context in numbers_with_context:
                    *contexts, num = context  # Unpack the context and the number
                    combined_context = " ".join(contexts).lower()  # Combine all context parts
                    
                    # Check if the number is actually a valid float and not part of a sentence
                    try:
                        num_value = float(num)
                    except ValueError:
                        continue
                    
                    if filter_value in combined_context:
                        filtered_numbers.append(num_value)
                
                if filtered_numbers:
                    total = sum(filtered_numbers)
                    return f"The total sum for {filter_condition} '{filter_value.capitalize()}' is {total}."
        
        # If the answer already contains a calculated total, return it directly
        total_match = re.search(r"total.*?(\d+(?:\.\d+)?)", answer, re.IGNORECASE)
        if total_match:
            return answer

        # If no filter condition or filtered results are found, and no total in the answer
        if numbers_with_context:
            total = sum(float(num) for *_, num in numbers_with_context if num.replace('.', '').isdigit())
            if total > 0:
                return f"The total sum of all values mentioned is {total}. {answer}"
    
    # If it's not an aggregation question or no aggregation was performed, return the answer as-is
    return answer

st.title("PDF Q&A Bot")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
qa = None

if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        try:
            db = process_pdf(uploaded_file)
            if db is not None:
                AGGREGATION_PROMPT = PromptTemplate.from_template("""
                Given the following conversation and a followup question, rephrase the followup question to be a standalone question.
                If the question requires any calculations or aggregations, please perform them and show your work.
                Make sure to filter the data based on any specified conditions (e.g., region, person, division, territory).
                Provide a step-by-step breakdown of your calculations.

                Chat History: {chat_history}
                Follow up Input: {question}

                Standalone question with calculations (if needed):
                """)

                llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
                qa = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=db.as_retriever(),
                    condense_question_prompt=AGGREGATION_PROMPT,
                    return_source_documents=True,
                    verbose=False
                )
                st.success("PDF processed successfully!")
            else:
                st.error("Failed to process PDF due to API errors. Please try again later.")
        except Exception as e:
            st.error(f"An error occurred while processing the PDF: {str(e)}")

    if qa is not None:
        chat_history = []

        st.header("Ask a question about your PDF")
        user_question = st.text_input("Enter your question:")

        if st.button("Get Answer"):
            if user_question:
                with st.spinner("Getting answer..."):
                    try:
                        result = qa({"question": user_question, "chat_history": chat_history})
                        processed_answer = post_process_aggregation(user_question, result['answer'])
                        
                        chat_history.append((user_question, processed_answer))
                        
                        st.write(f"**Question:** {user_question}")
                        st.write(f"**Answer:** {processed_answer}")
                    except OpenAIError as e:
                        if "RateLimitError" in str(e):
                            st.error("Rate limit exceeded while processing your question. Please wait a moment and try again.")
                        else:
                            st.error(f"An error occurred: {str(e)}")
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
            else:
                st.error("Please enter a question.")
    else:
        st.error("PDF processing failed. Please try uploading the file again or try a different PDF.")
else:
    st.info("Please upload a PDF file to get started.")

