
from langchain.embeddings.openai import OpenAIEmbeddings
import  streamlit as st
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor 



# loading PDF, DOCX and TXT files as LangChain Documents
def load_document(file):
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data


def chunk_data(data, chunk_size=256,chunk_overlap=25):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks


def create_embedding(chunks):
    embedding = OpenAIEmbeddings()
    vectore_store = Chroma.from_documents(chunks,embedding)
    return vectore_store

def ask_and_get_answer1(vector_store, q, k=5):
    from langchain.chains import RetrievalQA
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    answer = chain.invoke(q)
    return answer['result']

def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']



if __name__ == "__main__":
    import os
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)
    os.environ["OPENAI_API_KEY"]="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"

    st.subheader("Policy RAG")
    with st.sidebar:
        uplod_file = st.file_uploader("Upload a file",type=['pdf','docx','txt'])
        add_Data = st.button("Add Data",on_click=clear_history)


        if uplod_file and add_Data:
            with st.spinner("Reading, Chunking and Embedding"):
                bytes_Data = uplod_file.read()
                file_name = os.path.join("./",uplod_file.name)
                with open(file_name,"wb") as f:
                    f.write(bytes_Data)
                
                data = load_document(file_name)
                chunks = chunk_data(data,chunk_size=256)
                vector_store = create_embedding(chunks)
                st.session_state.vs = vector_store
                st.success("Fiile uploaded,chunked and embedded successfully")


    q=st.text_input("Please Enter the question about your File!")
    answer=None
    if q:
        if 'vs' in st.session_state:
            vector_store = st.session_state.vs   
            answer = ask_and_get_answer1(vector_store,q,k=3)      
            #print(answer)   
            st.text_area("LLM Answer" , value=answer) 

    
            st.divider()
    
            if "history" not in st.session_state:
                st.session_state.history = ''
            value = f'Q : {q}\nA: {answer}'
            st.session_state.history = f'{value} \n {"-"* 100} \n {st.session_state.history}'
            h=st.session_state.history
            st.text_area(label='Chat history',value=h,key='history',height=400)


            


