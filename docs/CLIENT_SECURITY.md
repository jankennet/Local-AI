# Client-side secret storage — rules for whatever connects to this server

The server issues two things a client must hold onto: an **API key**
(shared secret, set once via `LLM_API_KEY`) and a **session_id** (per-
conversation token, returned by `POST /sessions`). Both act as bearer
credentials — anyone holding either can act as that client.

## Never do this

- `localStorage.setItem("api_key", ...)` or `sessionStorage.setItem(...)`
  in a browser-based client. Any script that ever executes on that page
  (an XSS bug, a malicious extension, a compromised dependency) can read
  browser storage in full. This holds even on a trusted home LAN — the
  risk isn't the network, it's the browser tab.
- Committing the key to a repo, `.env` file that gets synced/backed up
  publicly, or embedding it directly in client-side JS source.

## Do this instead

**CLI / desktop / mobile-native clients:**
Store credentials in a local config file outside the web/JS layer, with
restrictive permissions:

```python
# ~/.config/llm-client/credentials.json, chmod 600
import os, json, stat

CONFIG_DIR = os.path.expanduser("~/.config/llm-client")
CONFIG_PATH = os.path.join(CONFIG_DIR, "credentials.json")

def save_credentials(api_key: str, session_id: str):
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump({"api_key": api_key, "session_id": session_id}, f)
    os.chmod(CONFIG_PATH, stat.S_IRUSR | stat.S_IWUSR)  # 0600 — owner read/write only
```

On mobile, use the OS-provided secure storage instead of a plain file:
Keychain on iOS, Keystore/EncryptedSharedPreferences on Android.

**If you eventually build a browser-based UI:**
Don't let the browser hold the API key at all. Put a small trusted
backend in front of it (can run on the same box) that holds the key
server-side, and have the browser authenticate to *that* backend using
an `HttpOnly`, `Secure`, `SameSite=Strict` cookie — which JavaScript
cannot read even if the page is compromised. The browser never sees the
raw API key or a token it could leak.

## Rotation

Since the key lives in one env var server-side, rotating it is:
`export LLM_API_KEY="<new value>"` + restart the server + update the
config file on each client. Worth doing if you ever suspect a device
(e.g. a phone) was lost or compromised.
