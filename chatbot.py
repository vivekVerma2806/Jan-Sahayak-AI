import os
import concurrent.futures
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Use the fastest practical model first and only fall back when needed.
GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
]

RETRIEVAL_TIMEOUT = 20   # seconds before giving up on vector search


def retrieve_relevant_chunks(vector_store, query: str, k: int = 3):
    """
    Retrieves the top k most relevant chunks along with their similarity scores.
    Uses a thread-based timeout so it never hangs the UI indefinitely.
    """
    def _search():
        try:
            return vector_store.similarity_search_with_relevance_scores(query, k=k)
        except Exception:
            # Fallback: plain similarity search without scores
            docs = vector_store.similarity_search(query, k=k)
            return [(doc, 0.5) for doc in docs]

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_search)
            return future.result(timeout=RETRIEVAL_TIMEOUT)
    except concurrent.futures.TimeoutError:
        raise RuntimeError(
            f"Vector database search timed out after {RETRIEVAL_TIMEOUT}s. "
            "Check your API key and internet connection."
        )
    except Exception as e:
        raise RuntimeError(f"Error during document retrieval: {e}")

def generate_answer(query: str, retrieved_chunks: list, google_api_key: str) -> dict:
    """
    Generates a response from Google Gemini strictly grounded in the retrieved chunks.
    If the context is insufficient, returns: "The information is not available in the uploaded documents."
    Automatically retries with fallback models if rate limits are hit.
    """
    fallback_response = "The information is not available in the uploaded documents."
    
    # If no chunks were returned, we can't answer the question
    if not retrieved_chunks:
        return {
            "answer": fallback_response,
            "sources": []
        }
    
    # 1. Format context and collect metadata for the user interface
    context_parts = []
    sources_info = []
    
    for i, (doc, score) in enumerate(retrieved_chunks):
        # Format the score as a percentage confidence value (clamp between 0 and 100)
        confidence = max(0.0, min(1.0, score)) * 100
        
        # Source filename
        source_name = os.path.basename(doc.metadata.get('source', 'Unknown Document'))
        page_num = doc.metadata.get('page', 0) + 1
        
        # Context block passed to LLM
        context_parts.append(
            f"--- [SOURCE {i+1}]: {source_name} (Page {page_num}) ---\n"
            f"{doc.page_content}"
        )
        
        # Details displayed in Streamlit UI
        sources_info.append({
            "id": i + 1,
            "content": doc.page_content,
            "source": source_name,
            "page": page_num,
            "score": round(confidence, 2)
        })
        
    context_text = "\n\n".join(context_parts)
    
    # 2. Construct prompt restricting Gemini to the provided context
    system_prompt = (
        "You are an expert AI assistant specializing in Government Schemes.\n"
        "You must answer the user's question STRICTLY based on the provided document context below.\n\n"
        f"--- CONTEXT START ---\n"
        f"{context_text}\n"
        f"--- CONTEXT END ---\n\n"
        "Strict Guidelines:\n"
        "1. Answer the question using ONLY the facts explicitly mentioned in the context above.\n"
        "2. Do NOT use any pre-existing training knowledge, external facts, or assumptions to answer.\n"
        "3. If the context does not contain the answer, or is unrelated to the question, you MUST answer EXACTLY with: \n"
        "   \"The information is not available in the uploaded documents.\"\n"
        "   Do not add any explanations, greetings, or other words if this is the case.\n"
        "4. Be objective, factual, and direct. Do not say things like 'based on the context' or 'according to source 1'."
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ]
    
    # 3. Try each model in the fallback chain
    last_error = None
    for model_name in GEMINI_MODELS:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=google_api_key,
                temperature=0.0,
                request_timeout=15,
                max_output_tokens=512,
            )

            # Wrap in a thread timeout so a hanging API never blocks the UI
            def _invoke():
                return llm.invoke(messages)

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_invoke)
                try:
                    response = future.result(timeout=15)
                except concurrent.futures.TimeoutError:
                    print(f"{model_name} timed out, trying next model...")
                    continue

            content = response.content

            # Parse list-based content (some versions of the package return structured blocks)
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        text_parts.append(part["text"])
                    elif hasattr(part, "text"):
                        text_parts.append(part.text)
                    elif isinstance(part, str):
                        text_parts.append(part)
                answer = "".join(text_parts).strip()
            else:
                answer = str(content).strip()

            # 4. Enforce exact fallback message if the LLM generated a generic refusal
            lower_answer = answer.lower()
            refusal_keywords = [
                "not mentioned", "not found", "does not contain", "no information",
                "not provide", "not available in the provided", "cannot answer",
                "insufficient information", "do not have information"
            ]

            if any(keyword in lower_answer for keyword in refusal_keywords) or len(answer) < 5:
                answer = fallback_response

            return {
                "answer": answer,
                "sources": sources_info
            }

        except Exception as e:
            last_error = e
            error_str = str(e).upper()
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in error_str or "RATE" in error_str or "404" in str(e) or "NOT_FOUND" in error_str:
                print(f"Error on {model_name} ({type(e).__name__}), trying next model...")
                continue
            else:
                return {
                    "answer": f"⚠️ Gemini API error: {str(e)}",
                    "sources": []
                }

    return {
        "answer": f"⚠️ All Gemini models timed out or are rate-limited. Please wait a moment and try again.\n\n_Details: {str(last_error)}_",
        "sources": []
    }

