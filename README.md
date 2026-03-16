## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Sudharma-11/Agentic-AI-Workshop.git
   ```

2. Install the required dependencies:
   ```
   pip install -r AgenticAIrequirements.txt
   ```
   pip install -r AgenticAIrequirements.txt
   ```

3. Create a `.env` file in the root directory and add your Google Gemini API Key:
   ```
   GOOGLE_API_KEY="your_api_key_here"
   ```
## Usage

1. Run the Streamlit application:
   ```
   streamlit run app.py
   ```

2. Open a web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`)

3. Interact with the Research Assistant to query topics and discover insights.

## Project Structure

- `app.py`: Streamlit web application for the user interface which contains the agent declaration
- `tools.py`: Initialization of tools (search, wikipedia, save) for the Agent.
- `AgenticAIrequirements.txt`: List of Python dependencies
- `agent.py`: CLI version of the agent script. This script will help you understand how to build an agent using the langchain framework.
