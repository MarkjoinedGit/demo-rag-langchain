from langchain_community.llms.ctransformers import CTransformers
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

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
    prompt = PromptTemplate(template = template, input_variables = ['question'])
    return prompt

def create_simple_chain(llm, prompt):
    llm_chain = LLMChain(llm = llm, prompt = prompt)
    return llm_chain

template = '''<|im_start|>system
Bạn là một trợ lí AI hữu ích. Hãy trả lời người dùng một cách chính xác.
<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant'''

llm = load_model()
prompt = create_prompt_template(template)
llm_chain = create_simple_chain(llm, prompt)

question = 'Mục đích của việc đạo tạo kỹ sư ngành Công nghệ thông tin là gì?'
output = llm_chain.invoke({'question': question})
print(output)




