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

2. **Install Poetry** (if you haven't already)

```bash
 poetry install
```

4. **Install and start Ollama**
- Follow the installation instructions at [Ollama's website](https://ollama.ai)
- Start the Ollama server
- Pull your preferred model (e.g., `ollama pull llama2`)

## Running the App 🏃‍♂️

1. **Activate the poetry environment and run program**
```bash
poetry shell
streamlit run datalens/app.py
```


3. **Access the app**
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