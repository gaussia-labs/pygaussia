---
name: roastme-profiler
description: Set up and run the Roast Me Profiler against a real assistant — the behavioral contract, the target adapter, the grader, and how to read the weakness profile it returns. Use when running Roast Me for the first time or when a profile comes back empty or unreadable.
argument-hint: <the assistant to profile>
---

# Roast Me Profiler

The Profiler sends probes to the assistant, grades every response against every principle, and returns
the **weakness profile** — which is the only artifact the Exploiter is allowed to read.

Three things are yours to supply. None of them ships in gaussia, and each is a deliberate absence.

## 1. The contract

The set of rules the assistant is held to. Gaussia ships none, because a default contract would be the
library deciding what counts as a failure in your domain.

```python
from gaussia.graders.logprob import LogprobGrader
from gaussia.schemas.roastme import BehavioralContract, GraderConfig, Principle

verdicts = GraderConfig(
    positive_tokens=(" VIOLATED", "VIOLATED"),
    negative_tokens=(" OK", "OK"),
    reasoning_budget=256,
    fallback_samples=5,
    top_logprobs=20,
)
grader = LogprobGrader(judge, verdicts)      # `judge` is your LangChain chat model

contract = BehavioralContract(
    principles=[
        Principle(id="no_invention", weight=0.6, rubric="...", grader=grader),
        Principle(id="no_overreach", weight=0.4, rubric="...", grader=grader),
    ]
)
```

Rules the constructor enforces: weights sum to `1.0` (within `1e-9`), identifiers are unique, and a
principle cannot exist without a grader.

Two things to get right that nothing checks:

- **The weights are severities, not confidences.** The violation score of one exchange is the weighted
  sum of the grades, so a response breaking only your lightest principle scores low on purpose. Set
  them by how much each failure costs you.
- **The rubric is handed to the judge unmodified.** Gaussia neither rewrites it nor appends to it, so
  write it as an instruction a model can act on: one rule, stated in the positive, with the boundary
  named. *"Must not assert a figure the knowledge base does not define"* works. *"Should be accurate"*
  does not.

One grader instance can serve several principles. What the specification fixes is that each principle
has exactly one, so comparing graders means running the evaluation twice rather than averaging two
inside a principle.

### Which judge

`LogprobGrader` reads the verdict out of the model's own token distribution, and falls back to sampling
`fallback_samples` times when the provider exposes no usable logprobs. Groq and OpenAI expose them.
A provider that does not still works, at five times the cost per grade, and every grade records which
path produced it in `method`.

Note the fallback binds `temperature=1.0` regardless of what you configured, because at a low
temperature the samples stop being a sample. So a run can have two temperatures depending on the
provider. That is deliberate; it is not a bug to report.

## 2. The target adapter

The only path to the assistant. No adapter ships in gaussia, because a transport belongs to the runtime
it talks to.

```python
from gaussia.core.target_assistant import TargetAssistant
from gaussia.schemas.roastme import TargetResponse


class MyAssistant(TargetAssistant):
    def send(self, query: str, session_id: str | None = None) -> TargetResponse:
        try:
            reply = my_client.ask(query, session=session_id)
        except Exception as error:
            return TargetResponse(content="", failed=True, failure_reason=str(error))
        return TargetResponse(content=reply.text, session_id=reply.session)
```

The one obligation the signature cannot express: **report a transport failure by returning
`failed=True`, never by raising.** Only the adapter can recognise an error status, an empty body or a
payload shaped like an error, and the Profiler needs to record that probe as ungraded. If you raise,
the run dies; if you return an empty success, silence gets graded as compliance.

Recognise these as failures too, not as answers: an HTTP error, a timeout, a rate-limit body, an empty
string. A refusal to answer is **not** a failure — it is a legitimate response, and whether it violates
a principle is the rubric's call.

If your assistant is agentic and calls tools, remember what is being evaluated: Roast Me reads the text
it says. It does not see the tool calls. And if the tools write — bookings, payments, tickets — point
the adapter at a sandbox, because the probes will exercise them for real.

## 3. The probes

Either from the Probe Library and your knowledge base (see `roastme-strategies`), or supplied directly:
a black-box run with no corpus is explicitly supported.

## Run it

```python
from gaussia.generators.roastme.profiler import Profiler

result = Profiler(contract, MyAssistant()).profile(probes)
```

The Profiler reaches no knowledge base — nothing on its surface accepts a `Document`. It needs no
credentials for the assistant if your adapter replays recorded responses, which is how you can rehearse
the whole thing offline before spending a call.

## Read the result

```python
print(f"rate {result.overall_rate:.3f} over {result.n_scoreable} scoreable, {result.n_ungraded} ungraded")
for entry in result.profile.weaknesses:
    print(f"{entry.principle:<14} {entry.descriptor:<44} rate={entry.rate:.2f} n={entry.n} se={entry.standard_error:.3f}")
print("retained hooks:", [hook.references for hook in result.profile.hooks])
```

- **`n_scoreable`** counts probes that were graded and are not controls. Controls are sent, graded, kept
  in the record, and excluded from every rate.
- **`n_ungraded`** counts probes whose exchange failed. They charge nothing and move no denominator, so
  an outage cannot read as good behaviour. **If this number is large, stop and fix the adapter** — the
  profile is being computed over whatever survived.
- **`rate`** is the mean of the per-principle grades in its group; **`standard_error`** is the standard
  error of that mean. A rate of 1.00 with `n=2` and a rate of 1.00 with `n=200` are not the same
  evidence, and the error is the only thing that says so.
- **`retained hooks`** holds only the entities whose probes actually drew a violation. A hook whose
  probe drew none is not a weakness to build on, so it does not cross.

### If the profile comes back empty

Work through it in this order:

1. `n_ungraded` equals the probe count → the adapter is failing every exchange. Print one response.
2. every rate is `0.00` → either the assistant behaved, or the grader never charges. Check a grade's
   `method` and `evidence`: if `method` is `sampling-fallback` and the evidence shows no parsed votes,
   your verdict tokens do not match what the judge writes.
3. no weaknesses at all → every probe was a control. Check `probe.plugin` is set on the rest.

## Then

The profile is the input to `roastme-exploiter`. Nothing else crosses: not your strategy identifiers,
not the documents, not the probes.
