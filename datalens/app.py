import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from langchain.llms import Ollama
from langchain import PromptTemplate, LLMChain

# Set Streamlit page configuration
st.set_page_config(page_title='CSV Analyzer', layout='wide')

# Initialize session state
if 'uploaded_files' not in st.session_state:
    st.session_state['uploaded_files'] = {}
if 'file_tabs' not in st.session_state:
    st.session_state['file_tabs'] = {}


def ai_agent_interaction(query, model_name, temperature, selected_df):
    """
    Interact with the AI agent to process user queries.
    """
    # Initialize the Ollama LLM
    llm = Ollama(model=model_name, base_url='http://localhost:11434', temperature=temperature)
    
    # Extract column information
    columns_info = ', '.join([f"'{col}'" for col in selected_df.columns])
    
    template = PromptTemplate(
        input_variables=["query", "columns_info"],
        template="""
You are an AI assistant specialized in data manipulation and analysis. The DataFrame 'df' has the following columns: {columns_info}. Perform the task described in the query using only this DataFrame. Do not redefine 'df' or use hardcoded data. Output only the Python code that achieves the task using pandas.

Query: {query}

Python code:"""
    )
    llm_chain = LLMChain(llm=llm, prompt=template)
    response = llm_chain.run(query=query, columns_info=columns_info)
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
    st.title('CSV Analyzer')
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

# Main content area
st.header('Data Viewer and Selector')
if st.session_state['uploaded_files']:
    file_names = list(st.session_state['uploaded_files'].keys())
    tabs = st.tabs(file_names)
    for idx, file_name in enumerate(file_names):
        df = st.session_state['uploaded_files'][file_name]
        with tabs[idx]:
            st.subheader(f'Data from {file_name}')
            # Column selection
            selected_columns = st.multiselect(f'Select Columns ({file_name})', df.columns.tolist(), key=f'select_columns_{file_name}')
            # Row selection
            selected_rows = st.multiselect(f'Select Rows ({file_name})', df.index.tolist(), key=f'select_rows_{file_name}')
            # Highlight selected rows and columns
            def highlight_selection(x):
                df_styled = pd.DataFrame('', index=df.index, columns=df.columns)
                if selected_rows:
                    for row in selected_rows:
                        df_styled.loc[row, :] = 'background-color: #ffffb3'  # Light yellow
                if selected_columns:
                    for col in selected_columns:
                        df_styled.loc[:, col] = 'background-color: #b3d1ff'  # Light blue
                if selected_rows and selected_columns:
                    for row in selected_rows:
                        for col in selected_columns:
                            df_styled.loc[row, col] = 'background-color: #ffcccb'  # Light red
                return df_styled

            styled_df = df.style.apply(highlight_selection, axis=None)
            st.dataframe(styled_df)

            # Store the selected DataFrame and its file name in session state
            if selected_rows and selected_columns:
                st.session_state['selected_df'] = df.loc[selected_rows, selected_columns]
            elif selected_rows:
                st.session_state['selected_df'] = df.loc[selected_rows, :]
            elif selected_columns:
                st.session_state['selected_df'] = df.loc[:, selected_columns]
            else:
                st.session_state['selected_df'] = df
            st.session_state['selected_df_file_name'] = file_name

# AI Agent Interaction
if user_query and model_name and check_ollama_server():
    st.header('AI Agent Response')
    if 'selected_df' in st.session_state:
        selected_df = st.session_state['selected_df']
        response = ai_agent_interaction(user_query, model_name, model_temperature, selected_df)
        st.write("AI Agent generated code:")
        st.code(response)
        # Attempt to execute the generated code
        try:
            local_vars = {'df': selected_df.copy(), 'pd': pd, 'px': px}
            exec(response, {}, local_vars)
            # Update the DataFrame if it has been modified
            new_df = local_vars.get('df')
            if new_df is not None and not selected_df.equals(new_df):
                # Update the DataFrame in session state
                file_name = st.session_state.get('selected_df_file_name')
                if file_name:
                    st.session_state['uploaded_files'][file_name] = new_df
                    st.success(f'DataFrame {file_name} updated successfully.')
                else:
                    st.error('Could not determine which file to update.')
            # Display plot if generated
            if 'fig' in local_vars:
                st.plotly_chart(local_vars['fig'])
        except Exception as e:
            st.error(f'Failed to execute AI agent response: {e}')
    else:
        st.warning('Please select data from a CSV file for the AI agent to analyze.')
else:
    st.warning('Please enter a query and ensure the Ollama server is running.')

# End of app.py
