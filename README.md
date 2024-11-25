# DataLens 🔍 - Your AI-Powered CSV Analysis Companion 📊

DataLens is an educational tool that helps you analyze CSV files using natural language queries and AI assistance! Perfect for students, data enthusiasts, and educators looking to explore data interactively. The tool accepts english prompts and outputs sql queries to find data within provided sources.

## Features ✨

- 🤖 AI-powered data analysis using Ollama models
- 📁 Support for multiple CSV files
- 🎯 Column and row selection capabilities
- 🌡️ Adjustable AI temperature settings
- 🎨 Highlighted data selection

## Prerequisites 🛠️

- Python 3.10 or higher
- Poetry package manager
- Ollama installed and running locally

## Setup Guide 🚀

1. **Clone the repository**
```bash
git clone <url>
cd datalens
```

2. **install requirements**
```bash
# Create virtual environment
python -m venv venv

# Activate it (on Windows)
venv\Scripts\activate
# OR on Unix/MacOS
source venv/bin/activate

# Install only the required packages
pip install streamlit pandas requests ollama duckdb
```

3. **Run and Access the app**
```bash
streamlit run app.py
```
- Open your browser and navigate to `http://localhost:8501`

## Usage Example 💡

1. Upload your CSV file(s)
2. Select columns and rows of interest
3. Choose your preferred Ollama model
4. Ask questions in natural language like:
   - "Show me a bar chart of total sales by category"
   - "Calculate the average price by product category"
   - "Find the top 5 customers by order value"

## Dependencies 📦

- Python 3.10+
- Streamlit
- Pandas
- Langchain
- Plotly
- Poetry

## License 📄

Use it as you wish! 🎉

## Contributing 🤝

Feel free to open issues and pull requests to help improve DataLens!