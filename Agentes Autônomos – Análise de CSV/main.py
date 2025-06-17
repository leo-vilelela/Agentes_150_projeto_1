import os
import streamlit as st
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
from dotenv import load_dotenv
from zipfile import ZipFile

# Unzip CSV files
# loading the temp.zip and creating a zip object
try:
    with ZipFile("202401_NFs.zip", 'r') as zObject:
        # Extracting all the members of the zip 
        # into a specific location.
        zObject.extractall(path="data")
except Exception:
    st.error("Não foi possível descompactar os arquivos CSV.")
    st.stop()

# Load env vars (if using .env)
load_dotenv()

# Setup API key
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not openai_api_key:
    st.error("Por favor escreva sua OpenAI API key no aquivo .env.")
    st.stop()

os.environ["OPENAI_API_KEY"] = openai_api_key

# Load the CSV once
@st.cache_data
def load_merged_data():
    invoices = pd.read_csv("data/202401_NFs_Cabecalho.csv")
    items = pd.read_csv("data/202401_NFs_Itens.csv")

    # Merge on invoice_id
    df_merged = pd.merge(items, invoices, on="CHAVE DE ACESSO", how="left")

    return df_merged
    
try:
    df = load_merged_data()
except Exception:
    st.error("Não foi possível ler os arquivos CSV.")
    st.stop()

llm = OpenAI(temperature=0)
agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)

# UI
st.title("🧠 Agente para análise de CSV")
context = '''
You are specialist in Brazilian invoices.
Are trained using two CSV files. 
Based on this file elaborate answers on Brazilian Portuguese for followed question.
Use maximum 100 words if you can uderstand the question. Reply with: 
"Não entendi. Pode refazer a pergunta?"
Question:{PERGUNTA}
'''
query = st.text_input("Faça sua pergunta sobre os dados dos arquivos CSV:")

if query:
    with st.spinner("Processando..."):
        try:
            agent_query = context.format(PERGUNTA=query)
            response = agent.run(agent_query)
            st.success("Resposta:")
            st.write(response)
        except Exception as e:
            st.error(f"Error: {e}")
