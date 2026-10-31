# Guardrail flows

Every path through the guardrail code in `rampart.guardrails`: the signature index the input guard and
the tool guard share, the three guard nodes and `refuse`. Each node is a plain function over the graph
state and writes one `guardrail_log` entry per decision, with the scores that produced it. How the
graph wires the nodes together (agent, tools, approval) is drawn in [AGENT_FLOWS.md](AGENT_FLOWS.md).

Colours: green passes, red blocks or withholds, amber records a reason to withhold.

## Signature scoring

### Scoring one piece of text

`SignatureIndex._best` scores a token sequence against every signature and keeps the best.

```mermaid
flowchart TD
    A["Features of the text<br/>unigrams + adjacent bigrams"]
    B["BM25 score against each signature"]
    C["Divide by the signature's self-score<br/>1.0 = as close as the signature is to itself"]
    D{"Shares a bigram<br/>with the signature?"}
    E["Cap the score at 0.5"]
    F["Keep the score"]
    G["Best signature wins<br/>score, signature, family"]
    A --> B --> C --> D
    D -- no --> E --> G
    D -- yes --> F --> G
```

- The cap stops one common word ("rules", "instructions") spread across a benign sentence from acting
  as a wildcard: without a shared bigram a signature can't score above 0.5, below every threshold.
- `SignatureIndex.load` rejects a signature with fewer than two tokens, so every signature has a bigram
  and the cap always applies when none is shared.
- Tokens are lowercase words of two or more characters, minus stopwords, with plurals stemmed
  ("rules" → "rule"), identically for signatures and text.

### Finding hits in a text

`SignatureIndex.scan(text, threshold)` finds the passages that score above a threshold, with character
offsets so they can be removed.

```mermaid
flowchart TD
    A["Tokenise the text"]
    B{"More than 24 tokens?"}
    C["One window: the whole text"]
    D["Windows of 24 tokens,<br/>starting every 12"]
    E["Score each window"]
    F{"Window above<br/>the threshold?"}
    G["Drop the window"]
    H["Score each sentence<br/>in the window"]
    I{"Any sentence above<br/>the threshold?"}
    J["Hits: those sentences"]
    K["Hit: the whole window"]
    L["Drop duplicate spans<br/>from overlapping windows"]
    A --> B
    B -- no --> C --> E
    B -- yes --> D --> E
    E --> F
    F -- no --> G
    F -- yes --> H --> I
    I -- yes --> J --> L
    I -- no --> K --> L
```

`SignatureIndex.score(text)` is `scan(text, threshold=-1)` and returns the best hit. With a threshold
of −1 every window passes, and any sentence that matches any signature at all counts as above it, so the
score is the best sentence score; a window's own score is used only when no sentence in it matches
anything. An empty text scores 0.

## Input guard and refuse

`Guards.input_guard` decides whether the analyst's question reaches the agent. Two layers: the
signature score, then the LLM classifier.

```mermaid
flowchart TD
    A["score(user_query)"]
    B{"Above injection_threshold<br/>0.70?"}
    C["LlmClassifier.classify"]
    P["route = agent"]
    R["refuse()<br/>refusal message, withheld = true"]
    A --> B
    B -- "yes: classifier skipped" --> R
    B -- no --> C
    C -- SAFE --> P
    C -- "INJECTION, or any error" --> R
    class P pass
    class R block
    classDef pass fill:#E1F5EE,stroke:#0F6E56,color:#085041
    classDef block fill:#FAECE7,stroke:#993C1D,color:#712B13
```

| Outcome | `route` | Log | What the user sees |
|---|---|---|---|
| Signature hit | `blocked` | `rule_hit: true`, `llm_reason: "skipped: layer 1 fired"` | Refusal naming "signature match" |
| Classifier hit | `blocked` | `llm_hit: true` with the classifier's reason | Refusal naming "classifier" |
| Classifier error or timeout | `blocked` | `llm_hit: true`, `llm_error` set (fails closed) | Refusal naming "classifier" |
| Passed | `agent` | `rule_hit: false`, `llm_hit: false` | Nothing yet; the agent runs |

`refuse` reads the last log entry to name the layer that fired, and sets both `delivered` and `withheld`.

## Tool guard

`Guards.tool_guard` runs after every tool step, once per tool message the step produced (the trailing
`ToolMessage`s). It removes injected passages before the agent reads them, and records which tools
answered and which chunks were retrieved, for the output guard.

```mermaid
flowchart TD
    A["Next tool message<br/>from the last step"]
    B{"Envelope parses<br/>with status ok?"}
    C["Add the tool to tools_called<br/>query_private_database: keep the chunk refs"]
    D["scan(text, tool_guard_threshold 0.70)"]
    E{"Any hits?"}
    F["Message unchanged<br/>logged with hits = 0"]
    G["Merge overlapping spans,<br/>replace each with a removal marker"]
    H["New ToolMessage with the same id<br/>replaces the original"]
    A --> B
    B -- yes --> C --> D
    B -- "error or unparsed" --> D
    D --> E
    E -- no --> F
    E -- yes --> G --> H
    class F pass
    class H block
    classDef pass fill:#E1F5EE,stroke:#0F6E56,color:#085041
    classDef block fill:#FAECE7,stroke:#993C1D,color:#712B13
```

- Error and unparsed results are scanned too (an error message can carry injected text), but they don't
  count as a tool that answered, so a later citation of that tool is withheld by the output guard.
- A hit removes the matching sentence when one scores above the threshold, otherwise the whole 24-token
  window (see [Finding hits in a text](#finding-hits-in-a-text)).
- The replacement keeps the original message id, so the `add_messages` reducer overwrites the original
  instead of appending.
- Each message gets a log entry with its status, the number of hits, the best score, the signature
  families and up to two examples of the removed text.

## Output guard

`Guards.output_guard` checks the agent's draft answer in a fixed order. Two steps change the text
without blocking it, three can add a reason to withhold it, and any reason withholds the answer.

```mermaid
flowchart TD
    A["Draft answer"]
    B["Redact identifiers<br/>IPs and internal hostnames to pseudonyms,<br/>e-mails to [email]"]
    C["Soften absolute assurance<br/>hedged, or reworded if its clause negates it"]
    D{"Any [Source: …] citation?"}
    D1["Append the citation note"]
    E{"Every cited tool ran?<br/>Documents cited only if<br/>query_private_database ran?"}
    E1["Reason: cites tools<br/>that did not run"]
    F{"Verbatim overlap with the<br/>retrieved chunks above 35%?"}
    F1["Reason: verbatim overlap<br/>with internal documents"]
    G["Add the disclaimer<br/>if not already present"]
    H{"LlmJudge verdict SAFE?"}
    H1["Reason: the judge's reasons"]
    I{"Any reason collected?"}
    W["Withhold: message listing<br/>the reasons, plus the disclaimer"]
    OK["Deliver the checked text"]
    A --> B --> C --> D
    D -- no --> D1 --> E
    D -- yes --> E
    E -- no --> E1 --> F
    E -- yes --> F
    F -- yes --> F1 --> G
    F -- no --> G
    G --> H
    H -- "UNSAFE, or any error" --> H1 --> I
    H -- yes --> I
    I -- yes --> W
    I -- no --> OK
    class OK pass
    class W block
    class E1,F1,H1 reason
    classDef pass fill:#E1F5EE,stroke:#0F6E56,color:#085041
    classDef block fill:#FAECE7,stroke:#993C1D,color:#712B13
    classDef reason fill:#FAEEDA,stroke:#854F0B,color:#633806
```

- The judge always runs, last, on the text the user would see, even when an earlier step already found
  a reason to withhold. It fails closed: an error or timeout is the verdict `ERROR`, which withholds.
- The citation check reads tool names from `[Source: …]`. A citation that names anything else (a
  document title) counts as a document citation and needs `query_private_database` to have run.
- The overlap check is the share of the answer's 10-word sequences found verbatim in the chunks the tool
  guard recorded. With no recorded chunks, no chunk store, or an error fetching them, the ratio is 0 and
  the check passes; a fetch error is logged as `overlap_error`.
- When the draft came from the `final_answer` tool, the guard also answers that tool call, so the
  message history stays well formed.
- The log entry records the redaction counts, the softening changes, the cited and uncalled tools, the
  overlap ratio, whether the disclaimer was added, the judge's verdict and reasons, and `withheld_why`.
