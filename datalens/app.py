import streamlit as st
import pandas as pd
import requests
import ollama  
import duckdb  
import json
import asyncio
from typing import Dict, Any


# Set Streamlit page configuration
st.set_page_config(page_title='CSV Analyzer', layout='wide')

# Initialize session state
if 'uploaded_files' not in st.session_state:
    st.session_state['uploaded_files'] = {}
if 'selected_df' not in st.session_state:
    st.session_state['selected_df'] = None
if 'selected_df_file_name' not in st.session_state:
    st.session_state['selected_df_file_name'] = None

def execute_sql_query(query: str) -> str:
    """Execute SQL query against the selected dataframe."""
    try:
        if st.session_state['selected_df'] is None:
            return json.dumps({"error": "No dataframe selected"})
        
        con = duckdb.connect()
        con.register('selected_df', st.session_state['selected_df'])
        result_df = con.execute(query).df()
        return json.dumps(result_df.to_dict(orient='records'))
    except Exception as e:
        return json.dumps({"error": str(e)})

async def ai_agent_interaction(query: str, model_name: str, temperature: float) -> Dict[str, Any]:
    """Handle AI agent interaction using Ollama's tool calling."""
    client = ollama.AsyncClient()
    
    # Prepare context about the available data
    columns_info = ', '.join([f"'{col}'" for col in st.session_state['selected_df'].columns])
    context = f"""You are a data analysis assistant. You have access to a table 'selected_df' with the following columns: {columns_info}.
    Generate and execute SQL queries to answer user questions about this data. Always return both the SQL query and its results.
    """
    
    # Initialize conversation
    messages = [
        {'role': 'system', 'content': context},
        {'role': 'user', 'content': query}
    ]
    
    # First API call: Get SQL query from the model
    response = await client.chat(
        model=model_name,
        messages=messages,
        tools=[{
            'type': 'function',
            'function': {
                'name': 'execute_sql_query',
                'description': 'Execute SQL query against the selected dataframe',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'query': {
                            'type': 'string',
                            'description': 'SQL query to execute'
                        }
                    },
                    'required': ['query']
                }
            }
        }]
    )
    
    messages.append(response['message'])
    sql_results = None  # Add this line to store query results

    # Execute the SQL query if the model made a tool call
    if response['message'].get('tool_calls'):
        for tool in response['message']['tool_calls']:
            if tool['function']['name'] == 'execute_sql_query':
                query_args = tool['function']['arguments']
                function_response = execute_sql_query(query_args['query'])
                sql_results = json.loads(function_response)  # Add this line to parse results
                messages.append({
                    'role': 'tool',
                    'content': function_response,
                    'name': tool['function']['name']
                })
    
    # Get final response with analysis
    final_response = await client.chat(
        model=model_name,
        messages=messages,
    )
    
    return {
        'sql_query': response['message']['tool_calls'][0]['function']['arguments'].get('query') if response['message'].get('tool_calls') else None,
        'analysis': final_response['message']['content'],
        'sql_results': sql_results  
    }


def extract_sql_query(response):
    # Extract the SQL query from the function call in the response
    start = response.find('<function_call>')
    end = response.find('</function_call>')
    if start != -1 and end != -1:
        function_call = response[start:end]
        query_start = function_call.find('"query": "') + 10
        query_end = function_call.find('"}', query_start)
        return function_call[query_start:query_end]
    return None

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

# Handle AI agent interaction
if user_query and model_name and check_ollama_server():
    st.header('AI Agent Response')
    if st.session_state['selected_df'] is not None:
        with st.spinner('Analyzing data...'):
            result = asyncio.run(ai_agent_interaction(user_query, model_name, model_temperature))
            if result['sql_query']:
                st.subheader('Generated SQL Query')
                st.code(result['sql_query'], language='sql')
                
                if result['sql_results']:  # Add this section to display results
                    st.subheader('Query Results')
                    if isinstance(result['sql_results'], list):
                        st.dataframe(pd.DataFrame(result['sql_results']))
                    else:
                        st.write(result['sql_results'])
            
            st.markdown(result['analysis'])
    else:
        st.warning('Please select data from a CSV file for the AI agent to analyze.')
else:
    st.warning('Please enter a query and ensure the Ollama server is running.')
