from langchain_community.llms.ctransformers import CTransformers
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS



model_name = "model/vinallama-7b-chat_q5_0.gguf"

def load_model():
    llm = CTransformers(
        model = model_name,
        model_type = 'llama',
        max_new_token = 1024,
        temparature = 0.01
    )
    return llm

def create_prompt_template(template):
    prompt = PromptTemplate(template = template, input_variables = ['context', 'question'])
    return prompt

def create_qa(llm, prompt, db):
    llm_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = 'stuff',
        # k = 3 means the retriever will find maximum 3 related documents
        retriever = db.as_retriever(search_kwargs = {'k': 3}, max_token_limit=1024),
        return_source_documents = True,
        chain_type_kwargs = {'prompt': prompt}
    )
    return llm_chain

def read_from_vectordb():
    model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
    gpt4all_kwargs = {'allow_download': 'True'}
    embedding_model = GPT4AllEmbeddings(
        model_name=model_name,
        gpt4all_kwargs=gpt4all_kwargs
    )
    db = FAISS.load_local('vectorstore\db_faiss', embedding_model, allow_dangerous_deserialization = True)
    return db

db = read_from_vectordb()
llm = load_model()

template = """<|im_start|>system\nSử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời\n
    {context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""
prompt = create_prompt_template(template)

qa_chain = create_qa(llm, prompt, db)

#test
question = 'Trầm cảm sau sinh là gì?'
output = qa_chain.invoke({'query': question})
print(output)
