"""Provider-agnostic OpenAI-compatible chat client for the cross-model panel (open-weight models
served by Together / Fireworks / DeepInfra / OpenRouter — all expose POST /v1/chat/completions).
Schema-free JSON (the reranker parses the text itself), robust retry, and it PROBES whether the
model accepts temperature=0 (dropping it on the deprecation/400 pattern, recording temperature=None),
so decoding provenance is captured exactly like the claude/gemini clients.

Configure per provider via env, e.g.:
  OAI_BASE_URL=https://api.together.xyz/v1   OAI_API_KEY_ENV=TOGETHER_API_KEY
  OAI_BASE_URL=https://api.fireworks.ai/inference/v1   OAI_API_KEY_ENV=FIREWORKS_API_KEY
  OAI_BASE_URL=https://openrouter.ai/api/v1   OAI_API_KEY_ENV=OPENROUTER_API_KEY
Model string is passed verbatim (e.g. "Qwen/Qwen3-14B", "deepseek-ai/DeepSeek-V4-Flash").
"""
from __future__ import annotations
import os, time, json, ssl, urllib.request, urllib.error

# urllib (unlike the anthropic/google SDKs) has no bundled CA store on this
# Python.framework build, so verify against certifi's bundle when available.
try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _SSL_CTX = None


class OAIClient:
    def __init__(self, model, base_url=None, api_key_env=None, max_tokens=8192):
        self.model_name = model
        self.base_url = (base_url or os.getenv("OAI_BASE_URL", "")).rstrip("/")
        key_env = api_key_env or os.getenv("OAI_API_KEY_ENV", "OPENAI_API_KEY")
        self.api_key = os.getenv(key_env)
        self.max_tokens = max_tokens
        self.temperature = 0                      # probed; set to None if the model rejects it
        self.json_mode = True                     # response_format=json_object; dropped if rejected
        self.no_think = True                      # disable reasoning-mode CoT (parity w/ closed models,
                                                  # which run direct-answer); dropped if provider rejects it.
                                                  # Reasoning models otherwise burn the whole token budget
                                                  # on hidden CoT and truncate the ranking (false "repair").
        self.last_usage_metadata = {}
        self.last_model = None
        if not self.base_url:
            raise RuntimeError("Set OAI_BASE_URL (or pass base_url) for the open-model provider.")
        if not self.api_key:
            raise RuntimeError(f"Set {key_env} in the environment for the open-model provider.")

    def _post(self, kw):
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(kw).encode(),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",
                     "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"},
            method="POST")
        with urllib.request.urlopen(req, timeout=180, context=_SSL_CTX) as r:
            return json.loads(r.read().decode())

    def generate(self, prompt):
        kw = {"model": self.model_name, "max_tokens": self.max_tokens,
              "messages": [{"role": "user", "content": prompt}]}
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        if self.json_mode:
            kw["response_format"] = {"type": "json_object"}   # forces valid JSON -> low repair
        if self.no_think:
            kw["chat_template_kwargs"] = {"enable_thinking": False}   # off -> model emits the ranking, not 22k of CoT
        last = None
        for attempt in range(6):
            try:
                resp = self._post(kw)
                self.last_model = resp.get("model")
                u = resp.get("usage") or {}
                self.last_usage_metadata = {"prompt_token_count": u.get("prompt_tokens", 0),
                                            "candidates_token_count": u.get("completion_tokens", 0)}
                return resp["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                body = ""
                try:
                    body = e.read().decode()[:300]
                except Exception:
                    pass
                lb = body.lower()
                if e.code == 400 and "temperature" in lb and "temperature" in kw:
                    kw.pop("temperature"); self.temperature = None; continue   # model rejects temp -> drop, retry
                if e.code == 400 and ("response_format" in lb or "json" in lb) and "response_format" in kw:
                    kw.pop("response_format"); self.json_mode = False; continue  # model rejects json mode -> drop
                if e.code == 400 and ("chat_template" in lb or "enable_thinking" in lb) and "chat_template_kwargs" in kw:
                    kw.pop("chat_template_kwargs"); self.no_think = False; continue  # provider rejects the flag -> drop
                last = e; time.sleep(5 * (attempt + 1))
            except Exception as e:
                last = e; time.sleep(5 * (attempt + 1))
        raise last
