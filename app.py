import os 
from groq import Groq 
from langchain_community.document_loaders import PyPDFLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS 
from langchain.prompts import PromptTemplate
from langchain_groq.chat_models import ChatGroq
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import gradio as gr 


# Loading the GROQ API KEY
load_dotenv()
groq_api_key= Groq(api_key=os.getenv("GROQ_API_KEY"))

# Defining the LLM
llm= ChatGroq(model_name="llama3-70b-8192") 

# Defining the prompt template
template= """
            You are a helpful PDF Assistant.
            When asked questions, you answer them based on the given PDF only.
            Do not make up answers.
            If you don't know the answer just say I don't know.
            
            Context:{context}
            Question:{question}

          """
          
prompt_template= PromptTemplate(template=template, input_variables=['context','question'])          

qa=None

# Defining the function that is called when a file is uploaded
def handle_upload(pdf_file):
    global qa
    
    pdf_name= pdf_file.name
    
    loader= PyPDFLoader(pdf_name)
    document= loader.load()
    
    text_splitter= RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap= 50)
    text= text_splitter.split_documents(document)
    
    embedding= FastEmbedEmbeddings(model="BAAI/bge-small-en-v1.5")
    
    db= FAISS.from_documents(text, embedding)
    
    retriever= db.as_retriever(search_type='similarity', kwargs={'k':4})
    
    qa= RetrievalQA.from_chain_type(
        llm=llm,
        retriever= retriever,
        return_source_documents=True,
        chain_type='stuff',
        chain_type_kwargs={'prompt':prompt_template}
    )
    
    return "PDF Uploaded Successfully !!!"

# Function that is called when a question is asked.
def chatbot(user_question, history):
    answer= qa(user_question)['result']
    history.append((user_question, answer))
    
    return history, history
        

# User Interface Using Gradio
with gr.Blocks(theme=gr.themes.Soft()) as working_app:
    gr.Markdown(
        """
        <div style="text-align: center; padding: 20px;">
            <h1>📄 PDF Assistant Bot</h1>
            <p>Ask questions and get accurate answers from your uploaded PDF files.</p>
            <p><b>Powered by LLaMA3 + LangChain + FAISS</b></p>
            <hr style="margin-top: 10px;">
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            upload = gr.File(
                label="📤 Upload your PDF",
                file_types=[".pdf"],
                interactive=True,
                
            )
            status = gr.Textbox(
                label="📌 Upload Status",
                placeholder="No file uploaded yet...",
                interactive=False,
                show_label=True,
                lines=2
            )
            assistant_tip = gr.Markdown(
                "🧠 <i>Tip: Upload a PDF first, then ask your question below!</i>",
                elem_id="tip-text"
            )

        with gr.Column(scale=2, min_width=500):
            output_box = gr.Chatbot(
                label="💬 Assistant Chat",
                height=420,
                bubble_full_width=False,
                show_copy_button=True
            )

    with gr.Row():
        with gr.Column(scale=4):
            user_input = gr.Textbox(
                placeholder="Type your question here and press Enter...",
                label="📝 Your Question",
                lines=1
            )
        

    
    state = gr.State([])

    
    upload.change(fn=handle_upload, inputs=[upload], outputs=[status])
    user_input.submit(fn=chatbot, inputs=[user_input, state], outputs=[output_box, state])
    
    
       
if __name__=="__main__":
    working_app.launch()        