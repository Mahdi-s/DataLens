import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from langchain.llms import Ollama
from langchain import PromptTemplate, LLMChain
import duckdb  # Added for SQL query execution

# Set Streamlit page configuration
st.set_page_config(page_title='CSV Analyzer', layout='wide')

# Initialize session state
if 'uploaded_files' not in st.session_state:
    st.session_state['uploaded_files'] = {}
if 'selected_df' not in st.session_state:
    st.session_state['selected_df'] = None
if 'selected_df_file_name' not in st.session_state:
    st.session_state['selected_df_file_name'] = None

def ai_agent_interaction(query, model_name, temperature, selected_df):
    """
    Interact with the AI agent to translate user queries into SQL.
    """
    # Initialize the Ollama LLM
    llm = Ollama(model=model_name, base_url='http://localhost:11434', temperature=temperature)
    
    # Extract column information
    columns_info = ', '.join([f"'{col}'" for col in selected_df.columns])
    table_name = 'selected_df'
    
    template = PromptTemplate(
        input_variables=["query", "columns_info", "table_name"],
        template="""
You are an AI assistant specialized in data manipulation and analysis. The table '{table_name}' has the following columns: {columns_info}. Perform the task described in the query using only this table. Do not use hardcoded data. Output only the SQL query that achieves the task.

Query: {query}

SQL query:"""
    )
    llm_chain = LLMChain(llm=llm, prompt=template)
    response = llm_chain.run(query=query, columns_info=columns_info, table_name=table_name)
    return response

# Function to get available Ollama models
def get_available_ollama_models():
    """
    Retrieve a list of available Ollama models from the local Ollama server.
    """
    try:
        response = requests.get('http://localhost:11434/api/tags')
        if response.status_code == 200:
            models = response.json()['models']
            return [model['name'] for model in models]
        return []
    except (requests.exceptions.RequestException, KeyError):
        return []

def check_ollama_server():
    """
    Check if the Ollama server is running locally.
    """
    try:
        requests.get('http://localhost:11434/api/tags', timeout=5)
        return True
    except requests.exceptions.RequestException:
        return False

# Sidebar components
with st.sidebar:
    st.title('Data Lens 🔍')

    st.header('AI Agent Query')
    user_query = st.text_input('Enter your query for the AI agent:')
    model_temperature = st.slider('Model Temperature', min_value=0.0, max_value=1.0, value=0.7)
    st.header('Model Selection')
    if check_ollama_server():
        available_models = get_available_ollama_models()
        if available_models:
            model_name = st.selectbox('Select Ollama Model', available_models)
        else:
            st.warning('No models found. Please ensure Ollama models are loaded.')
            model_name = None
    else:
        st.error('Ollama server is not running. Please start Ollama to use the AI agent.')
        model_name = None

        st.header('File Upload')
    uploaded_files = st.file_uploader('Upload CSV files', type='csv', accept_multiple_files=True)
    # Manage uploaded files
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state['uploaded_files']:
                df = pd.read_csv(uploaded_file)
                st.session_state['uploaded_files'][uploaded_file.name] = df

    if st.session_state['uploaded_files']:
        file_names = list(st.session_state['uploaded_files'].keys())
        selected_file = st.selectbox('Select File for Analysis', file_names)
        df = st.session_state['uploaded_files'][selected_file]

        st.subheader(f'Select Data from {selected_file}')
        # Column selection
        selected_columns = st.multiselect(f'Select Columns ({selected_file})', df.columns.tolist(), key=f'select_columns_{selected_file}')
        # Row selection
        selected_rows = st.multiselect(f'Select Rows ({selected_file})', df.index.tolist(), key=f'select_rows_{selected_file}')
        # Store the selected DataFrame in session state
        if selected_rows and selected_columns:
            st.session_state['selected_df'] = df.loc[selected_rows, selected_columns]
        elif selected_rows:
            st.session_state['selected_df'] = df.loc[selected_rows, :]
        elif selected_columns:
            st.session_state['selected_df'] = df.loc[:, selected_columns]
        else:
            st.session_state['selected_df'] = df
        st.session_state['selected_df_file_name'] = selected_file

# Main content area
st.header('Data Viewer')
if st.session_state['uploaded_files']:
    file_names = list(st.session_state['uploaded_files'].keys())
    tabs = st.tabs(file_names)
    for idx, file_name in enumerate(file_names):
        df = st.session_state['uploaded_files'][file_name]
        with tabs[idx]:
            st.subheader(f'Data from {file_name}')
            # Get selected columns and rows for this file
            selected_columns = st.session_state.get(f'select_columns_{file_name}', df.columns.tolist())
            selected_rows = st.session_state.get(f'select_rows_{file_name}', df.index.tolist())
            # Highlight selected rows and columns
            def highlight_selection(x):
                df_styled = pd.DataFrame('', index=df.index, columns=df.columns)
                if selected_rows:
                    for row in selected_rows:
                        if row in df.index:
                            df_styled.loc[row, :] = 'background-color: #e6ffcc'  # Light yellow
                if selected_columns:
                    for col in selected_columns:
                        if col in df.columns:
                            df_styled.loc[:, col] = 'background-color: #b3d1ff'  # Light blue
                if selected_rows and selected_columns:
                    for row in selected_rows:
                        for col in selected_columns:
                            if row in df.index and col in df.columns:
                                df_styled.loc[row, col] = 'background-color: #ffcccb'  # Light red
                return df_styled

            styled_df = df.style.apply(highlight_selection, axis=None)
            st.dataframe(styled_df)

# AI Agent Interaction
if user_query and model_name and check_ollama_server():
    st.header('AI Agent Response')
    if 'selected_df' in st.session_state and st.session_state['selected_df'] is not None:
        selected_df = st.session_state['selected_df']
        with st.spinner('Generating SQL query...'):
            response = ai_agent_interaction(user_query, model_name, model_temperature, selected_df)
        st.write("AI Agent generated SQL query:")
        st.code(response)
        # Attempt to execute the generated SQL query
        try:
            con = duckdb.connect()
            con.register('selected_df', selected_df)
            result_df = con.execute(response).df()
            st.write("Result of the SQL query:")
            st.dataframe(result_df)
        except Exception as e:
            st.error(f'Failed to execute SQL query: {e}')
    else:
        st.warning('Please select data from a CSV file for the AI agent to analyze.')
else:
    st.warning('Please enter a query and ensure the Ollama server is running.')
