from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS


#declare variables 
pdf_data_path = 'data'
vector_db_path = 'vectorstore/db_faiss'

def create_db_from_raw_text():
    raw_text = '''Phân bố thời gian học tập: 4(3:1:8)
    Điều kiện tiên quyết: Không
    Các môn học trước: Nhập môn lập trình
    Tóm tắt nội dung học phần:
    Học phần này gồm hai phần chính:
    Phần “Toán rời rạc” trang bị cho người học những kiến thức cơ bản về logic mệnh đề, logic vị từ, suy diễn logic, tập hợp, ánh xạ, quan hệ tương đương, quan hệ thứ tự, dàn và đại số Bool. Cung cấp cho người học kiến thức và kỹ năng trong việc phân tích, nhìn nhận vấn đề, trong việc xác định công thức đa thức tối tiểu bằng phương pháp biểu đồ Karnaugh.
    Phần “Lý thuyết đồ thị” trang bị sự hiểu biết về các lĩnh vực ứng dụng của lý thuyết đồ thị, cung cấp kiến thức nền tảng về lý thuyết đồ thị ứng dụng trong tin học. Cung cấp các thuật toán, kỹ thuật và kỹ năng lập trình các giải thuật trong lý thuyết đồ thị.'''

    #splitter
    text_splitter =CharacterTextSplitter(
        separator="/n",
        chunk_size=512,
        chunk_overlap=50,
        length_function=len
    )

    chunks = text_splitter.split_text(raw_text)

    #embeding
    model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
    gpt4all_kwargs = {'allow_download': 'True'}
    embedding_model = GPT4AllEmbeddings(
        model_name=model_name,
        gpt4all_kwargs=gpt4all_kwargs
    )

    #vectorstore
    db = FAISS.from_texts(chunks, embedding=embedding_model)
    db.save_local(vector_db_path)
    return db

def create_db_from_file():
    loader = DirectoryLoader(path=pdf_data_path, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()

    #splitter
    documents_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = documents_splitter.split_documents(documents)

    #embedding
    model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
    gpt4all_kwargs = {'allow_download': 'True'}
    embedding_model = GPT4AllEmbeddings(
        model_name=model_name,
        gpt4all_kwargs=gpt4all_kwargs
    )

    #vector_store
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(vector_db_path)
    return db

create_db_from_file()