# Gaussia Agentic Lambda

AWS Lambda function for evaluating AI agent conversations with pass@K and tool correctness metrics.

## Description

This Lambda function uses the Gaussia `Agentic` metric to evaluate complete agent conversations. Each Dataset represents one complete conversation. A conversation is correct only if ALL its interactions are correct. It provides probabilistic evaluations using:

### Formulas

```
pass@k = 1 - C(n-c, k) / C(n, k)   # Probability of ≥1 correct conversation
pass^k = (c/n)^k                    # Probability of all k conversations correct
```

Where:
- **n** = total conversations evaluated
- **c** = fully correct conversations (all interactions correct)
- **k** = number of conversations to attempt (user-configurable)

### Metrics Provided

1. **pass@K** (0.0-1.0): Probability of ≥1 correct conversation when attempting k conversations
2. **pass^K** (0.0-1.0): Probability of all k conversations being fully correct
3. **Conversation Correctness**: A conversation is correct only if ALL interactions are correct
4. **Tool Correctness** (per interaction): Evaluates proper tool usage across four dimensions:
   - **Selection** (25%): Were the correct tools chosen?
   - **Parameters** (25%): Were the correct parameters passed?
   - **Sequence** (25%): Were tools used in the correct order? (if required)
   - **Utilization** (25%): Were tool results properly used in the final answer?

5. **Aggregated Metrics**: Overall agent performance across all conversations (recommended for evaluation)

The metric uses an LLM judge to evaluate answer correctness and direct dictionary comparison for tool correctness.

## Invoke URL

```
<DEPLOY_TO_GET_URL>
```

## Supported LLM Providers

| Provider | class_path |
|----------|-----------|
| Groq | `langchain_groq.chat_models.ChatGroq` |
| OpenAI | `langchain_openai.chat_models.ChatOpenAI` |
| Google Gemini | `langchain_google_genai.chat_models.ChatGoogleGenerativeAI` |
| Ollama | `langchain_ollama.chat_models.ChatOllama` |

## Test Example

### Using Groq

```bash
curl -s -X POST "<INVOKE_URL>/run" \
  -H "Content-Type: application/json" \
  -d '{
    "connector": {
      "class_path": "langchain_groq.chat_models.ChatGroq",
      "params": {
        "model": "llama-3.3-70b-versatile",
        "api_key": "your-groq-api-key",
        "temperature": 0.0
      }
    },
    "datasets": [
      {
        "session_id": "conversation_001",
        "assistant_id": "agent_v1",
        "language": "english",
        "context": "Math calculator conversation",
        "conversation": [
          {
            "qa_id": "q1_interaction1",
            "query": "What is 5 + 3?",
            "assistant": "The result is 8.",
            "ground_truth_assistant": "5 + 3 equals 8",
            "agentic": {
              "tools_used": [
                {
                  "tool_name": "calculator",
                  "parameters": {"operation": "add", "a": 5, "b": 3},
                  "result": 8,
                  "step": 1
                }
              ],
              "final_answer_uses_tools": true
            },
            "ground_truth_agentic": {
              "expected_tools": [
                {
                  "tool_name": "calculator",
                  "parameters": {"operation": "add", "a": 5, "b": 3},
                  "step": 1
                }
              ],
              "tool_sequence_matters": false
            }
          },
          {
            "qa_id": "q1_interaction2",
            "query": "What is 10 * 2?",
            "assistant": "10 times 2 is 20.",
            "ground_truth_assistant": "20"
          }
        ]
      },
      {
        "session_id": "conversation_002",
        "assistant_id": "agent_v1",
        "language": "english",
        "context": "Simple Q&A conversation",
        "conversation": [
          {
            "qa_id": "q2_interaction1",
            "query": "What is the capital of France?",
            "assistant": "The capital of France is Paris.",
            "ground_truth_assistant": "Paris"
          }
        ]
      }
    ],
    "config": {
      "threshold": 0.7,
      "tool_threshold": 0.75,
      "k": 3,
      "verbose": false
    }
  }' | jq
```

### Using OpenAI

```bash
curl -s -X POST "<INVOKE_URL>/run" \
  -H "Content-Type: application/json" \
  -d '{
    "connector": {
      "class_path": "langchain_openai.chat_models.ChatOpenAI",
      "params": {
        "model": "gpt-4o-mini",
        "api_key": "your-openai-api-key"
      }
    },
    "datasets": [...],
    "config": {
      "threshold": 0.7
    }
  }' | jq
```

## Request Format

```json
{
  "connector": {
    "class_path": "langchain_groq.chat_models.ChatGroq",
    "params": {
      "model": "llama-3.3-70b-versatile",
      "api_key": "your-api-key",
      "temperature": 0.0
    }
  },
  "datasets": [
    {
      "session_id": "conversation_001",
      "assistant_id": "agent_v1",
      "language": "english",
      "context": "Complete conversation context",
      "conversation": [
        {
          "qa_id": "interaction_1",
          "query": "First user question",
          "assistant": "Agent's first response",
          "ground_truth_assistant": "Expected first response",
          "agentic": {
            "tools_used": [
              {
                "tool_name": "tool_name",
                "parameters": {},
                "result": "tool_result",
                "step": 1
              }
            ],
            "final_answer_uses_tools": true
          },
          "ground_truth_agentic": {
            "expected_tools": [
              {
                "tool_name": "tool_name",
                "parameters": {},
                "step": 1
              }
            ],
            "tool_sequence_matters": false
          }
        },
        {
          "qa_id": "interaction_2",
          "query": "Follow-up question",
          "assistant": "Agent's second response",
          "ground_truth_assistant": "Expected second response"
        }
      ]
    },
    {
      "session_id": "conversation_002",
      "assistant_id": "agent_v1",
      "language": "english",
      "context": "Another complete conversation",
      "conversation": [
        {
          "qa_id": "interaction_1",
          "query": "Different question",
          "assistant": "Different response",
          "ground_truth_assistant": "Expected response"
        }
      ]
    }
  ],
  "config": {
    "threshold": 0.7,
    "tool_threshold": 0.75,
    "tool_weights": {
      "selection": 0.25,
      "parameters": 0.25,
      "sequence": 0.25,
      "utilization": 0.25
    },
    "k": 3,
    "use_structured_output": false,
    "verbose": false
  }
}
```

### Connector Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `connector.class_path` | string | Yes | Full class path of the LangChain chat model |
| `connector.params` | object | Yes | Parameters passed to the chat model constructor |
| `connector.params.model` | string | Yes | Model name/identifier |
| `connector.params.api_key` | string | Yes | API key for the LLM provider |
| `connector.params.temperature` | float | No | Sampling temperature (default varies by provider) |

### Agentic-Specific Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `datasets` | array | Yes | List of datasets - each Dataset represents one complete conversation |
| `datasets[].session_id` | string | Yes | Unique conversation identifier |
| `datasets[].assistant_id` | string | Yes | Assistant/agent identifier |
| `datasets[].conversation` | array | Yes | List of interactions in the conversation |
| `datasets[].conversation[].qa_id` | string | Yes | Unique identifier for each interaction |
| `datasets[].conversation[].agentic` | object | No | Agent's actual tool usage for this interaction |
| `datasets[].conversation[].agentic.tools_used` | array | No | List of tools used by the agent |
| `datasets[].conversation[].agentic.final_answer_uses_tools` | boolean | No | Whether tool results are used in final answer |
| `datasets[].conversation[].ground_truth_agentic` | object | No | Expected tool usage for this interaction |
| `datasets[].conversation[].ground_truth_agentic.expected_tools` | array | No | List of expected tools |
| `datasets[].conversation[].ground_truth_agentic.tool_sequence_matters` | boolean | No | Whether tool order matters (default: false) |
| `config.threshold` | float | No | Answer correctness threshold (0.0-1.0, default: 0.7) |
| `config.tool_threshold` | float | No | Tool correctness threshold (0.0-1.0, default: 0.75) |
| `config.tool_weights` | object | No | Weights for tool correctness components (default: 0.25 each) |
| `config.k` | integer | No | K value for pass@K calculations (default: 3) |
| `config.use_structured_output` | boolean | No | Use LangChain structured output (default: false) |
| `config.verbose` | boolean | No | Enable verbose logging (default: false) |

## Response Format

```json
{
  "success": true,
  "per_conversation_metrics": [
    {
      "session_id": "conversation_001",
      "total_interactions": 3,
      "correct_interactions": 3,
      "is_fully_correct": true,
      "threshold": 0.7,
      "correctness_scores": [0.850, 0.920, 0.880],
      "correct_indices": [0, 1, 2],
      "tool_correctness_scores": [
        {
          "tool_selection_correct": 1.0,
          "parameter_accuracy": 1.0,
          "sequence_correct": 1.0,
          "result_utilization": 1.0,
          "overall_correctness": 1.0,
          "is_correct": true,
          "reasoning": "All tools used correctly"
        },
        null,
        {
          "tool_selection_correct": 1.0,
          "parameter_accuracy": 0.5,
          "sequence_correct": 1.0,
          "result_utilization": 1.0,
          "overall_correctness": 0.875,
          "is_correct": true,
          "reasoning": "Minor parameter deviation"
        }
      ]
    },
    {
      "session_id": "conversation_002",
      "total_interactions": 2,
      "correct_interactions": 1,
      "is_fully_correct": false,
      "threshold": 0.7,
      "correctness_scores": [0.920, 0.650],
      "correct_indices": [0],
      "tool_correctness_scores": [null, null]
    }
  ],
  "aggregated_metrics": {
    "total_conversations": 3,
    "fully_correct_conversations": 2,  // Conversations where ALL interactions correct
    "conversation_success_rate": 0.6667,  // c/n = 2/3
    "k": 3,  // K value for calculations
    "pass_at_k": 0.9630,  // 96.3% chance of ≥1 correct conversation in 3 attempts
    "pass_pow_k": 0.2963,  // 29.63% chance of all 3 conversations correct
    "interpretation": "inconsistent"  // reliable | inconsistent | functional | needs_improvement
  }
}
```

### Response Fields

#### Per-Conversation Metrics

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | string | Unique conversation identifier |
| `total_interactions` | integer | Number of interactions in the conversation |
| `correct_interactions` | integer | Number of correct interactions |
| `is_fully_correct` | boolean | True if ALL interactions are correct |
| `threshold` | float | Answer correctness threshold used |
| `correctness_scores` | float[] | Score for each interaction (0.0-1.0) |
| `correct_indices` | int[] | Indices of correct interactions |
| `tool_correctness_scores` | object[] | Tool evaluation per interaction (null if no tools used) |

#### Aggregated Metrics (Recommended)

| Field | Type | Description |
|-------|------|-------------|
| `total_conversations` | integer | Number of conversations evaluated (n) |
| `fully_correct_conversations` | integer | Conversations where ALL interactions correct (c) |
| `conversation_success_rate` | float | Percentage of fully correct conversations (c/n) |
| `k` | integer | K value for pass@K calculations |
| `pass_at_k` | float | Probability of ≥1 correct conversation in K attempts |
| `pass_pow_k` | float | Probability of all K conversations being correct |
| `interpretation` | string | Agent quality: `reliable` \| `inconsistent` \| `functional` \| `needs_improvement` |

## Error Responses

```json
{
  "success": false,
  "error": "Error message describing what went wrong"
}
```

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `No connector configuration provided` | Missing `connector` field | Add connector with class_path and params |
| `connector.class_path is required` | Missing class_path in connector | Specify the LLM class path |
| `Failed to create LLM connector` | Invalid class_path or params | Check class_path spelling and required params |
| `No datasets provided` | Missing or empty `datasets` | Add at least one dataset |
| `No qa_ids found in datasets` | Datasets have no conversations | Add conversations with qa_id |
| `Agentic evaluation failed` | Evaluation error | Check logs for details |

## View Logs

```bash
aws logs tail "/aws/lambda/gaussia-agentic" --follow --region us-east-1
```

## Deployment Commands

```bash
# Deploy (from examples/agentic/aws-lambda/)
./scripts/deploy.sh agentic us-east-1

# Update (rebuild and redeploy)
./scripts/update.sh agentic us-east-1

# Cleanup (remove all AWS resources)
./scripts/cleanup.sh agentic us-east-1
```

## Environment Variables

The Lambda function supports the following environment variables:

| Variable | Description |
|----------|-------------|
| `LLM_API_KEY` | Fallback API key if not provided in request |

## Understanding pass@K vs pass^K

These metrics use probabilistic formulas from research ([reference](https://www.philschmid.de/agents-pass-at-k-pass-power-k)):

```
pass@k = 1 - C(n-c, k) / C(n, k)   # Probability of ≥1 correct conversation
pass^k = (c/n)^k                    # Probability of all k conversations correct
```

- **pass@K** (0.0-1.0): Probability of getting *at least one* fully correct conversation when attempting k conversations. High values (>0.9) indicate the agent *can* complete conversations successfully.
  - Example: pass@3 = 0.92 → 92% chance of getting ≥1 fully correct conversation in 3 attempts
  - A conversation is fully correct only if ALL its interactions are correct

- **pass^K** (0.0-1.0): Probability that *all* k conversations are fully correct. High values (>0.7) indicate the agent is *consistent* and reliable.
  - Example: pass^3 = 0.15 → 15% chance of all 3 conversations being fully correct

### Interpretation

| pass@K | pass^K | Assessment |
|--------|--------|------------|
| >0.95 | >0.70 | ✅ **Reliable** - High success and consistency |
| >0.95 | <0.50 | ⚠️ **Inconsistent** - Can succeed but unreliable |
| 0.70-0.95 | any | 🔶 **Functional** - Works but could improve |
| <0.70 | any | ❌ **Needs Improvement** - Low success rate |

### Conversation-Level Evaluation

This metric evaluates **complete conversations** as units:
- Each Dataset = 1 complete conversation with multiple interactions
- A conversation is correct only if ALL interactions are correct
- This measures the agent's ability to maintain fully correct multi-turn conversations
- **n** = total conversations evaluated
- **c** = fully correct conversations
- **K** = user-configurable (how many conversations to attempt)

## Tool Correctness Evaluation

Tool correctness is evaluated across four dimensions:

1. **Selection (25%)**: Did the agent choose the right tools?
2. **Parameters (25%)**: Did the agent pass correct parameters to the tools?
3. **Sequence (25%)**: Did the agent use tools in the correct order? (only matters if `tool_sequence_matters=true`)
4. **Utilization (25%)**: Did the agent use tool results in the final answer?

The overall score is a weighted average. You can customize weights in `config.tool_weights`.

## Installation

```bash
# Install Gaussia with agentic support
pip install gaussia[agentic]

# Install LLM provider (choose one or more)
pip install langchain-groq        # For Groq
pip install langchain-openai      # For OpenAI
pip install langchain-google-genai # For Google Gemini
pip install langchain-ollama      # For Ollama
```

---

Built with [Gaussia](https://github.com/gaussia-labs/pygaussia) - AI Evaluation Framework
