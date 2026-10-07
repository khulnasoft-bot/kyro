# API Reference

## POST /v1/chat/completions

Request:

```json
{
  "model": "llama3",
  "messages": [{"role": "user", "content": "Hello"}],
  "prompt": null,
  "stream": false,
  "max_tokens": 64,
  "temperature": 1.0,
  "top_p": 1.0,
  "top_k": null,
  "response_format": {"type": "json_object"}
}
```

- `messages` or `prompt` is required (400 otherwise).
- `model` must equal the configured `--model-name` (400 otherwise).
- `max_tokens` must be within `1..=KYRO_MAX_TOKENS_CAP`; `temperature` in
  `[0,2]`; `top_p` in `(0,1]`; message count ≤ `KYRO_MAX_MESSAGES`; prompt
  size ≤ `KYRO_MAX_PROMPT_BYTES`.
- `stream: true` returns `text/event-stream` chunks
  (`chat.completion.chunk`), decoded incrementally, then a final chunk with
  `finish_reason: "stop"`.
- Errors return `{"error": {"message", "type"}}` with a 4xx status.

Response (non-streaming): standard `chat.completion` object with decoded
`choices[0].message.content`.

## GET /v1/models

Lists the served model in OpenAI list format.

## GET /health

`200 OK` plain text.

## GET /metrics

Prometheus text exposition: request counters by model label, queue depth,
prefix-cache hits/misses, token counters, TTFT/TBT histograms, KV-cache
usage.
