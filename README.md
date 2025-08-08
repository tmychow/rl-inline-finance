# AI Financial Assistant Demo

A demonstration of how reinforcement learning can be used to improve small language models for financial data queries. This project shows that while small local models are initially inaccurate and good remote models are too slow, RL fine-tuning on small models can bridge the gap.

## Features

- **Inline Text Completion**: Copilot-style suggestions for financial queries with grey text that appears as you type
- **Multi-tool Agent**: AI agent that can make multiple tool calls to retrieve financial data
- **Model Comparison**: Compare performance between local (Ollama) and remote (OpenAI) models
- **Evaluation Suite**: Upload CSV test cases and evaluate model accuracy
- **Activity Logging**: Real-time visualization of all tool calls made by the agent
- **Latency Tracking**: Monitor end-to-end request times for each model
- **Financial Data Viewer**: Browse financial fundamentals by ticker

## Architecture

### Backend
- FastAPI server with async support
- SQLite database for financial data storage
- Agent system with tool calling capabilities:
  - `get_metrics()`: List available financial metrics
  - `get_tickers()`: List available stock tickers
  - `get_value(metric, ticker, period)`: Retrieve specific financial values
  - `calculate(num1, num2, operation, duration)`: Perform calculations including CAGR
  - `return_answer(answer)`: Return the final completion

### Frontend
- React/TypeScript application
- Real-time inline text suggestions
- Tab to accept completions
- Activity log showing tool calls
- Evaluation interface with CSV upload
- Fundamentals data viewer

## Setup

1. Clone the repository
2. Set up Python environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Initialize the database:
   ```bash
   python backend/database.py
   python data/tiingo_loader.py
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Add your API keys to .env
   ```

5. Install frontend dependencies:
   ```bash
   cd frontend
   npm install
   ```

## Running the Application

1. Start the backend server:
   ```bash
   source venv/bin/activate
   python backend/main.py
   ```

2. In a new terminal, start the frontend:
   ```bash
   cd frontend
   npm start
   ```

3. Open http://localhost:3000 in your browser

## Usage

### Text Editor Mode
- Start typing financial queries like "Apple's revenue in"
- Wait for grey suggestion text to appear
- Press Tab to accept the suggestion
- View tool calls in the activity log
- Monitor latency in the metrics table

### Evaluation Mode
- Upload a CSV file with 'input' and 'ground_truth' columns
- Select models to evaluate
- Run batch evaluation to see accuracy scores
- View individual predictions with correctness indicators

### Fundamentals Viewer
- Select a ticker from the dropdown
- View all available financial metrics in a table format
- Data organized by time period

## Generating Test Data

Generate synthetic test cases:
```bash
python scripts/generate_synthetic_data.py
```
This creates:
- `data/evaluation_test_cases.csv`: 50 test cases for evaluation
- `data/training_test_cases.csv`: 200 test cases for training

The generator now covers cross-ticker comparisons, multi-metric ratios, and a
wider variety of no-completion prefixes that mention random tickers and metrics
without requiring a numeric answer. This results in more realistic benchmarking.

## Models Supported

- **OpenAI**: gpt-4o-mini (remote, fast, accurate)
- **Ollama**: qwen2.5:3b (local, can be fine-tuned)

## Future Work

- Implement RL fine-tuning pipeline for local models
- Add more financial metrics and calculations
- Support for real-time market data
- Enhanced evaluation metrics
- Model serving infrastructure for fine-tuned models

LLM-as-a-judge for correctness + always calling get_tickers/get_metrics first + efficiency penalty (fewer turns)

Other ideas:
1. Tool call validity - Penalize malformed tool calls or invalid arguments (e.g., requesting non-existent tickers/metrics)
2. Information completeness - Reward when the model retrieves all necessary data before attempting calculations (e.g., getting both revenue values before calculating growth)
3. Calculation accuracy - Separate reward for correct math operations when using the calculate tool, even if final answer is wrong
4. Redundancy penalty - Discourage repeated identical tool calls within the same conversation
