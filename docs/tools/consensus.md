# Consensus Tool - Multi-Model Perspective Gathering

**Get diverse expert opinions from multiple AI models on technical proposals and decisions**

The `consensus` tool orchestrates multiple AI models to provide diverse perspectives on your proposals, enabling structured decision-making through for/against analysis and multi-model expert opinions.

## Thinking Mode

**Default is `medium` (8,192 tokens).** Use `high` for complex architectural decisions or `max` for critical strategic choices requiring comprehensive analysis.

## Model Recommendation

Consensus tool uses extended reasoning models by default, making it ideal for complex decision-making scenarios that benefit from multiple perspectives and deep analysis.

## How It Works

The consensus tool orchestrates multiple AI models to provide diverse perspectives on your proposals:

1. **Assign stances**: Each model can take a specific viewpoint (supportive, critical, or neutral)
2. **Gather opinions in one call** (`mode: "parallel"`, the default): every model on the roster is consulted concurrently, each blinded to the others (it sees only your proposal and the context files), and the response carries all verdicts in `accumulated_responses` in roster order together with a `panel` summary (`requested`, `consulted`, `succeeded`, `failed`, `timed_out`, `skipped_by_host_policy`, `failed_models`, `deadline_seconds`)
3. **Synthesize results**: the calling agent combines all perspectives into a balanced recommendation and finishes with a `VERDICT: <approve|revise|reject>` line
4. **Natural language**: Use simple descriptions like "supportive", "critical", or "against" - the tool handles synonyms automatically

The legacy one-model-per-step loop is still available as `mode: "sequential"`: step 1 consults the first model, each later step (with the returned `continuation_id`) consults the next, and `next_step_required` tells the agent when to synthesize. When `mode` is omitted but the request is shaped like that loop (`total_steps > 1` or `next_step_required: true`), the server runs it sequentially so older callers keep working.

### Parallel mode semantics

- **Caller shape that works on every server version**: send `mode: "parallel"` together with the normal step fields (`step_number: 1`, `total_steps: <model count>`, `next_step_required: true`). A server with parallel support returns the whole panel and `next_step_required: false`; an older server ignores `mode` and runs the loop. Stop as soon as the response reports `consensus_complete` or `next_step_required: false`.
- **Partial panels are reported, never hidden**: a leg that errors or misses the deadline is returned as `status: "error"` / `"timed_out"` with its error text, `status` becomes `consensus_workflow_partial`, `consensus_complete` is `false`, and `next_steps` names the missing models. Only when *no* leg succeeds does the call fail (MCP `isError`) with the same `panel` and `accumulated_responses` in the error payload.
- **Panel deadline**: `CONSENSUS_PANEL_DEADLINE_S` (default `1500`) bounds the wait for the slowest leg. Legs still running at the deadline are marked `timed_out` and their tasks cancelled; the underlying provider call keeps running until its HTTP read timeout, so the cost is still incurred. Set it below your MCP client's tool timeout.
- **The schema advertises no `default` for `mode`** on purpose: a client that materialises schema defaults would otherwise send `mode: "parallel"` on a loop-shaped request and skip the inference. Omit `mode` and the server infers; send it and it wins. Every terminal error payload (all legs failed, step ≥ 2 on a parallel continuation, restore miss) carries `next_step_required: false` and `consensus_complete: false`, so a caller that loops only while `next_step_required` is true always stops.
- **Stale client schemas**: an MCP client that captured the tool list before the fork was installed may strip the unknown `mode` field; the loop-shaped request then runs sequentially and the response says so in `metadata.mode`. Restart the client session after a PAL redeploy and check `metadata.mode` in the first response.
- **One round per continuation**: calling again with `step_number >= 2` on a parallel continuation is rejected with an error that re-supplies the verdicts — do not re-run the panel. Start a new round with `step_number: 1` (optionally on the same `continuation_id`).
- **Response size**: a six-model panel returns six full verdicts (roughly 6 × 850 tokens) in a single response; run it from a subagent or an out-of-band wrapper rather than the main conversation.

## Watch In Action

The following is a hypothetical example designed to demonstrate how one consensus can be built upon another (via [continuation](../context-revival.md)). In this scenario, we start with a _blinded_ consensus, where one model is tasked with taking a **for** stance and another with an **against** stance. This approach allows us to see how each model evaluates a particular option relative to the alternative. We then conduct a second consensus — all initiated by a single prompt and orchestrated by Claude Code in this video — to gather each model’s final conclusions.

<div style="center">
  
  [PAL Consensus Debate](https://github.com/user-attachments/assets/76a23dd5-887a-4382-9cf0-642f5cf6219e)
  
</div>

## Example Prompts

**For/Against Analysis:**
```
Use pal consensus with flash taking a supportive stance and pro being critical to evaluate whether 
we should migrate from REST to GraphQL for our API
```

**Multi-Model Technical Decision:**
```
Get consensus from o3, flash, and pro on our new authentication architecture. Have o3 focus on 
security implications, flash on implementation speed, and pro stay neutral for overall assessment
```

**Natural Language Stance Assignment:**
```
Use consensus tool with gemini being "for" the proposal and grok being "against" to debate 
whether we should adopt microservices architecture
```

```
I want to work on module X and Y, unsure which is going to be more popular with users of my app. 
Get a consensus from gemini supporting the idea for implementing X, grok opposing it, and flash staying neutral
```

## Key Features

- **Stance steering**: Assign specific perspectives (for/against/neutral) to each model with intelligent synonym handling
- **Custom stance prompts**: Provide specific instructions for how each model should approach the analysis
- **Ethical guardrails**: Models will refuse to support truly bad ideas regardless of assigned stance
- **Unknown stance handling**: Invalid stances automatically default to neutral with warning
- **Natural language support**: Use terms like "supportive", "critical", "oppose", "favor" - all handled intelligently
- **Parallel panel (default)**: every model consulted concurrently in one call under a panel deadline; `mode: "sequential"` keeps the one-model-per-step loop
- **Focus areas**: Specify particular aspects to emphasize (e.g., 'security', 'performance', 'user experience')
- **File context support**: Include relevant files for informed decision-making
- **Image support**: Analyze architectural diagrams, UI mockups, or design documents
- **Conversation continuation**: Build on previous consensus analysis with additional rounds
- **Web search capability**: Enhanced analysis with current best practices and documentation

## Tool Parameters

- `step`: The proposal or question every model will see (required). In sequential mode, steps 2+ carry the agent's private notes instead
- `step_number` / `total_steps` / `next_step_required`: workflow position (required). Parallel mode answers with `1` / `1` / `false`
- `findings`: The agent's own analysis (required; never sent to the models)
- `mode`: `parallel` (default) or `sequential` — see *Parallel mode semantics* above
- `models`: List of model configurations with optional stance and custom instructions (required at step 1; at least two)
- `relevant_files`: Context files for informed analysis (absolute paths)
- `images`: Visual references like diagrams or mockups (absolute paths)
- `continuation_id`: Continue previous consensus discussions (required for sequential steps 2+)

Environment: `CONSENSUS_PANEL_DEADLINE_S` — parallel panel deadline in seconds (default `1500`).

## Model Configuration Examples

**Basic For/Against:**
```json
[
    {"model": "flash", "stance": "for"},
    {"model": "pro", "stance": "against"}
]
```

**Custom Stance Instructions:**
```json
[
    {"model": "o3", "stance": "for", "stance_prompt": "Focus on implementation benefits and user value"},
    {"model": "flash", "stance": "against", "stance_prompt": "Identify potential risks and technical challenges"}
]
```

**Neutral Analysis:**
```json
[
    {"model": "pro", "stance": "neutral"},
    {"model": "o3", "stance": "neutral"}
]
```

## Usage Examples

**Architecture Decision:**
```
"Get consensus from pro and o3 on whether to use microservices vs monolith for our e-commerce platform"
```

**Technology Migration:**
```
"Use consensus with flash supporting and pro opposing to evaluate migrating from MySQL to PostgreSQL"
```

**Feature Priority:**
```
"Get consensus from multiple models on whether to prioritize mobile app vs web dashboard development first"
```

**With Visual Context:**
```
"Use consensus to evaluate this new UI design mockup - have flash support it and pro be critical"
```

## Best Practices

- **Provide detailed context**: Include project constraints, requirements, and background
- **Use balanced stances**: Mix supportive and critical perspectives for thorough analysis
- **Specify focus areas**: Guide models to emphasize relevant aspects (security, performance, etc.)
- **Include relevant files**: Provide code, documentation, or specifications for context
- **Build on discussions**: Use continuation for follow-up analysis and refinement
- **Leverage visual context**: Include diagrams, mockups, or design documents when relevant

## Ethical Guardrails

The consensus tool includes built-in ethical safeguards:
- Models won't support genuinely harmful proposals regardless of assigned stance
- Unknown or invalid stances automatically default to neutral
- Warning messages for potentially problematic requests
- Focus on constructive technical decision-making

## When to Use Consensus vs Other Tools

- **Use `consensus`** for: Multi-perspective analysis, structured debates, major technical decisions
- **Use `chat`** for: Open-ended discussions and brainstorming
- **Use `thinkdeep`** for: Extending specific analysis with deeper reasoning
- **Use `analyze`** for: Understanding existing systems without debate
