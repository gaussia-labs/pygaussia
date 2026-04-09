# Agentic Metric - Jupyter Examples

Interactive notebooks demonstrating the Agentic metric for evaluating AI agent responses.

## Contents

| Notebook | Description |
|----------|-------------|
| `agentic.ipynb` | Complete Agentic metric demonstration with pass@K, pass^K, and tool correctness evaluation |
| `dataset_agentic.json` | Sample dataset with K=3 agent responses for testing |

## What is the Agentic Metric?

The Agentic metric evaluates AI agent responses across multiple attempts (K responses) and measures:

1. **pass@K**: At least one of K responses is correct (similarity >= threshold)
2. **pass^K**: All K responses are correct (measures consistency)
3. **Tool Correctness**: Evaluates proper tool usage across 4 dimensions:
   - **Selection (25%)**: Were the correct tools chosen?
   - **Parameters (25%)**: Were correct parameters passed?
   - **Sequence (25%)**: Were tools used in the correct order?
   - **Utilization (25%)**: Were tool results used in the final answer?

## Quick Start

### 1. Install Dependencies

```bash
pip install "gaussia[agentic]" langchain-groq jupyter
```

### 2. Get an API Key

You'll need a Groq API key (free tier available):
- Sign up at https://console.groq.com/
- Generate an API key

### 3. Launch Jupyter

```bash
cd examples/agentic/jupyter/
jupyter notebook agentic.ipynb
```

### 4. Follow the Notebook

The notebook will guide you through:
- Loading the example dataset (K=3 responses)
- Initializing the Agentic metric with a judge model
- Running the evaluation
- Analyzing pass@K vs pass^K results
- Visualizing tool correctness scores
- Customizing thresholds and weights

## Dataset Structure

The example dataset (`dataset_agentic.json`) contains:
- **3 agent responses** (K=3) for each question
- **3 different questions** testing various scenarios
- **Tool usage information** for each response

Example structure:
```json
[
  {
    "session_id": "agentic_eval_session",
    "assistant_id": "agent_response_1",
    "conversation": [
      {
        "qa_id": "q1_math",
        "query": "What is 15 + 27?",
        "assistant": "Using the calculator, I found that 15 + 27 = 42.",
        "ground_truth_assistant": "15 + 27 equals 42.",
        "agentic": {
          "tools_used": [
            {
              "tool_name": "calculator",
              "parameters": {"operation": "add", "a": 15, "b": 27},
              "result": 42,
              "step": 1
            }
          ],
          "final_answer_uses_tools": true
        },
        "ground_truth_agentic": {
          "expected_tools": [
            {
              "tool_name": "calculator",
              "parameters": {"operation": "add", "a": 15, "b": 27},
              "step": 1
            }
          ],
          "tool_sequence_matters": false
        }
      }
    ]
  }
]
```

## Key Concepts

### pass@K vs pass^K

- **pass@K**: Useful for evaluating if an agent *can* produce correct answers
  - Example: pass@3=true means at least 1 of 3 responses was correct
  - Good for measuring capability

- **pass^K**: Measures consistency and reliability
  - Example: pass^3=true means all 3 responses were correct
  - Good for production readiness assessment

### Tool Correctness Components

Each component is scored 0.0-1.0:

1. **Selection**: Compares `tools_used` names against `expected_tools` names
2. **Parameters**: Compares parameter dictionaries for exact matches
3. **Sequence**: Checks if tools were used in the correct order (only if `tool_sequence_matters=true`)
4. **Utilization**: Verifies `final_answer_uses_tools=true`

Overall score = weighted average of components (default: 0.25 each)

## Customization

### Custom Thresholds

```python
metrics = Agentic.run(
    MyRetriever,
    model=judge_model,
    threshold=0.8,        # Answer correctness threshold
    tool_threshold=0.7,   # Tool correctness threshold
)
```

### Custom Tool Weights

Emphasize certain aspects of tool usage:

```python
metrics = Agentic.run(
    MyRetriever,
    model=judge_model,
    tool_weights={
        "selection": 0.4,    # Emphasize tool selection
        "parameters": 0.3,   # Emphasize parameters
        "sequence": 0.2,
        "utilization": 0.1,
    }
)
```

## Use Cases

1. **Agent Reliability Testing**: Compare pass@K vs pass^K to understand consistency
2. **Tool Usage Analysis**: Identify which aspects of tool usage need improvement
3. **Model Comparison**: Compare different LLM models or prompting strategies
4. **Quality Benchmarking**: Track agent performance over time
5. **Production Readiness**: Use pass^K to determine if an agent is production-ready

## Example Results

From the notebook, you'll see results like:

```
Question ID: q1_math
K (responses): 3
Threshold: 0.7

Correctness scores: [0.92, 0.88, 0.90]
Correct indices: [0, 1, 2]

pass@3: True ✓  (at least 1 correct)
pass^3: True ✓  (all 3 correct)

Tool Correctness:
  Selection:    1.00
  Parameters:   1.00
  Sequence:     1.00
  Utilization:  1.00
  Overall:      1.00 ✓
```

## Supported LLM Providers

The Agentic metric works with any LangChain-compatible chat model:

```python
# Groq (recommended for speed)
from langchain_groq import ChatGroq
model = ChatGroq(model="llama-3.3-70b-versatile")

# OpenAI
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4o")

# Anthropic Claude
from langchain_anthropic import ChatAnthropic
model = ChatAnthropic(model="claude-3-5-sonnet-20241022")

# Ollama (local)
from langchain_ollama import ChatOllama
model = ChatOllama(model="llama3.1:70b")
```

## Troubleshooting

### "Module not found: gaussia"

Make sure you installed with the `agentic` extra:
```bash
pip install "gaussia[agentic]"
```

### "API key not provided"

Set your API key before running:
```python
import getpass
GROQ_API_KEY = getpass.getpass("Enter your Groq API key: ")
```

### "No datasets provided"

Ensure your retriever returns **multiple datasets with the same qa_id**:
- Same `qa_id` across datasets
- Different `assistant_id` for each response
- Minimum K=2 responses per question

## Next Steps

After exploring the notebook:

1. **Create Your Own Dataset**: Follow the structure in `dataset_agentic.json`
2. **Deploy to AWS Lambda**: See `../aws-lambda/` for deployment examples
3. **Integrate with CI/CD**: Automate agent evaluation in your pipeline
4. **Track Metrics Over Time**: Use pass@K and tool correctness as KPIs

## Additional Resources

- [Gaussia Documentation](https://github.com/gaussia-labs/pygaussia)
- [Groq Console](https://console.groq.com/)
- [LangChain Documentation](https://python.langchain.com/)

---

Built with [Gaussia](https://github.com/gaussia-labs/pygaussia) - AI Evaluation Framework
