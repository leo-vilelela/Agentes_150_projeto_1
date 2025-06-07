import os
import streamlit as st
import pandas as pd
#from langchain.agents import create_pandas_dataframe_agent
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
from dotenv import load_dotenv

# Load env vars (if using .env)
load_dotenv()

# Setup API key
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not openai_api_key:
    st.error("Please set your OpenAI API key in .env or Streamlit secrets.")
    st.stop()

os.environ["OPENAI_API_KEY"] = openai_api_key

# Load the CSV once
#@st.cache_data
#def load_data():
#    df = pd.read_csv("data/your_static_file.csv")
#    return df

@st.cache_data
def load_merged_data():
    invoices = pd.read_csv("data/202401_NFs_Cabecalho.csv")
    items = pd.read_csv("data/202401_NFs_Itens.csv")

    # Merge on invoice_id
    df_merged = pd.merge(items, invoices, on="CHAVE DE ACESSO", how="left")

    return df_merged
    

df = load_merged_data()
llm = OpenAI(temperature=0)
agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)

# UI
st.title("🧠 CSV Question Answering Agent")
query = st.text_input("Ask a question about your CSV data:")

if query:
    with st.spinner("Thinking..."):
        try:
            response = agent.run(query)
            st.success("Answer:")
            st.write(response)
        except Exception as e:
            st.error(f"Error: {e}")
