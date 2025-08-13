import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def get_groq_api_key():
    """
    Gets the Groq API key.
    Tries to get it from Streamlit's secrets first (for deployment).
    If that fails, it falls back to loading from a local .env file.
    """
    try:
        # This works only when deployed on Streamlit Community Cloud
        api_key = st.secrets["GROQ_API_KEY"]
        print("Loaded API key from Streamlit secrets.")
        return api_key
    except (FileNotFoundError, KeyError):
        # Fallback for local development
        print("Loading API key from local .env file.")
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        return api_key

# --- Simple Fallback Function ---
def generate_simple_explanation(class_name, confidence_score):
    """
    A simple, template-based fallback generator.
    """
    confidence_percent = confidence_score * 100
    if class_name == "PNEUMONIA":
        return (
            f"The model's analysis of the image highlights regions with patterns "
            f"suggestive of pneumonia, with a confidence of {confidence_percent:.1f}%. "
            f"The highlighted areas on the saliency map indicate the primary features "
            f"influencing this prediction. Further review by a specialist is recommended."
        )
    else:
        return (
            f"The model found no significant visual patterns typically associated with pneumonia, "
            f"predicting the scan as normal with a confidence of {confidence_percent:.1f}%. "
            f"Follow-up based on clinical symptoms is advised."
        )

# --- LLM-Powered Generator (with Saliency Map Legend & General Info) ---
def generate_llm_explanation(class_name, confidence_score):
    """
    Generates a clinical explanation using the Groq API and LangChain.
    It now includes a detailed, easy-to-understand legend for the saliency map colors
    and general educational information. Falls back to a simple template if the API call fails.
    """
    try:
        api_key = get_groq_api_key()
        
        if not api_key:
            print("Warning: GROQ_API_KEY not found. Using simple explanation.")
            return generate_simple_explanation(class_name, confidence_score)

        # Use a powerful, available model from Groq
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name="llama3-70b-8192",
            temperature=0.2 # Low temperature for factual, consistent output
        )

        # --- Enhanced Prompt for PNEUMONIA cases ---
        system_prompt = (
            "You are an expert AI assistant explaining a medical imaging diagnosis to a user. "
            "Your tone should be simple, confident, and clear. "
            "You will be given a diagnosis and a confidence score. Your response MUST be structured in four parts, exactly as follows, with bolded titles:\n\n"
    
            "1.  **Clinical Summary:** A short, professional summary. Use advisory language like 'suggestive of' or 'patterns consistent with'. Never give a definitive diagnosis.\n\n"
    
            "2.  **How to Read the Saliency Map:** An explanation of the colored heatmap overlay. You MUST use the following exact wording for the colors:\n"
            "    - **Bright Red/Yellow areas:** 'These are the pixels I examined most closely. They are the key reasons I decided this patient has pneumonia.'\n"
            "    - **Blue/Green areas:** 'These are pixels I noticed but ignored, as they were not important for my decision.'\n\n"
    
            "3.  **General Information on Pneumonia Management:** Provide simple, general educational guidance on how pneumonia is commonly managed, "
            "without implying a prescription or personal treatment plan. Mention options like rest, hydration, prescribed antibiotics (for bacterial pneumonia), "
            "and medical monitoring.\n\n"
    
            "4.  **Recommendation:** End with this exact sentence: 'Further review by a qualified specialist is recommended.'"
        )

        # --- Corrected and Enhanced Prompt for NORMAL cases ---
        if class_name == "NORMAL":
             system_prompt = (
                "You are an expert AI assistant explaining a medical imaging diagnosis to a user. Your tone should be simple, confident, and clear. "
                "You will be given a diagnosis and a confidence score. Your response MUST be structured in three parts, exactly as follows, with bolded titles:\n\n"
                
                "1.  **Clinical Summary:** A short, professional summary stating that the model did not find patterns consistent with pneumonia.\n\n"
                
                "2.  **How to Read the Saliency Map:** An explanation of the colored heatmap overlay. You MUST use the following exact wording for the colors:\n"
                "    - **Bright Red/Yellow areas:** 'These are the pixels I examined most closely to confirm that tell-tale signs of pneumonia were absent.'\n"
                "    - **Blue/Green areas:** 'These are pixels I noticed but ignored, as they were not important for my decision.'\n\n"
                
                "3.  **Recommendation:** End with this exact sentence: 'Further review by a qualified specialist is recommended.'"
            )
            
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", 
                 "The model's prediction is '{diagnosis}' with a confidence of {confidence:.1f}%. Please generate the response in the specified format.")
            ]
        )

        # Create the chain with a string output parser
        chain = prompt_template | llm | StrOutputParser()
        
        confidence_percent = confidence_score * 100
        response = chain.invoke({"diagnosis": class_name, "confidence": confidence_percent})
        
        return response

    except Exception as e:
        print(f"Error calling LLM: {e}. Using simple explanation as a fallback.")
        return generate_simple_explanation(class_name, confidence_score)