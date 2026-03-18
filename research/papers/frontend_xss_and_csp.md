---
topic: Frontend XSS and Content Security Policy
status: ready_for_implementation
priority: high
estimated_complexity: medium
researched_at: 2026-03-18T14:00:00Z
---

# Frontend XSS and Content Security Policy

## Problem Statement

The Agent Zero frontend (agent_zero.html) has multiple DOM-based XSS vectors and zero Content Security Policy enforcement. Server-generated content from LLM agents is injected via innerHTML without consistent sanitization. No CSP header exists anywhere in the codebase, and the entire frontend uses inline scripts and styles with no nonce or hash protections.

## Current State in Agent Zero

### innerHTML Injection Points (60+ total)

The frontend has 60+ innerHTML assignments. Most use `esc()` (line 4064) for server data, but several critical paths do NOT:

**Unescaped server data injected via innerHTML:**
- `agent_zero.html:2740` -- `thought` (LLM agent output) injected raw: `headerHtml += '<span class="agent-thought-text">${thought}</span>'` then `header.innerHTML = headerHtml` at line 2748
- `agent_zero.html:2739` -- `agentName` injected raw (from `_AGENT_DISPLAY_NAMES` lookup, lower risk but no defense-in-depth)
- `agent_zero.html:3274` -- `req.category` and `req.priority` injected without `esc()` in renderRequest()
- `agent_zero.html:3455` -- `g.status` injected into style attribute via string interpolation (CSS injection)

**ID injection into onclick handlers:**
- `agent_zero.html:3278` -- `req.id` in `onclick="approveRequest('${req.id}')"` -- if ID contains `'`, breaks out of handler
- `agent_zero.html:3459-3464` -- `g.id` in multiple onclick handlers (updateGoalStatus, deleteGoal)
- `agent_zero.html:3582-3590` -- `c.id` in multiple onclick handlers (updateCommitmentStatus, deleteCommitment)

**Note:** `renderMarkdown()` (line 2633) escapes `<`, `>`, `&` BEFORE markdown processing, so code blocks are safe. But the function returns HTML that is set via innerHTML at lines 2574, 2624, 3377.

### Missing Sanitization Library

No HTML sanitizer (DOMPurify or Trusted Types) is loaded. The `esc()` function (line 4064) and `_escDetail()` (line 2858) use `textContent`->`innerHTML` which is correct but applied inconsistently.

### No Content Security Policy

- Zero instances of `Content-Security-Policy` in the entire agent_zero/ codebase
- `agent_zero_server.py` sets no security headers on HTML responses
- All CSS is inline in a `<style>` tag (~4000 lines)
- All JS is inline in a `<script>` tag (~2200 lines)
- External script at line 7: `/assets/index-DDTFU3NR.js` (no integrity attribute)

### JWT Token Exposure

- `agent_zero.html:1940` -- JWT stored in `localStorage` (accessible to any JS on page, including XSS payloads)
- `agent_zero.html:2131` -- JWT passed in WebSocket URL query parameter: `ws://.../ws/chat?token=...` (logged in server access logs, browser history, proxy logs)

### Missing CSRF Protection

- All POST/DELETE fetch calls use `Authorization: Bearer` header but no CSRF token
- `agent_zero_server.py:187` -- CORS configured with `allow_headers=["*"]` (wildcard)
- No `SameSite` cookie attribute since tokens aren't in cookies

### Empty Catch Blocks

- `agent_zero.html:3293` -- `approveRequest()`: `catch {}` silently swallows errors
- `agent_zero.html:3308` -- `rejectRequest()`: same pattern
- `agent_zero.html:3445` -- goals loading: `catch {}` silently fails

## Industry Standard / Research Findings

### OWASP DOM-Based XSS Prevention (2025)

The OWASP DOM-based XSS Prevention Cheat Sheet states: "The best way to fix DOM-based XSS is to use the right output method (sink). If you want to use user input to write in a div, use `textContent` or `innerText`, NOT `innerHTML`." When HTML rendering is required, all content MUST pass through a sanitizer before innerHTML assignment.

**Source:** https://cheatsheetseries.owasp.org/cheatsheets/DOM_based_XSS_Prevention_Cheat_Sheet.html

### DOMPurify (cure53, 2025)

DOMPurify is the industry standard DOM-only XSS sanitizer (13k+ GitHub stars). It removes `<script>`, `onerror`, `javascript:` URLs, and other XSS vectors while preserving safe HTML. It works with any markup (HTML, MathML, SVG) and has zero dependencies. Current version: 3.x (2025).

**Source:** https://github.com/cure53/DOMPurify

**Research source:** Heiderich et al., "DOMPurify: Client-Side Protection Against XSS and Markup Injection", ESORICS 2017. https://link.springer.com/chapter/10.1007/978-3-319-66399-9_7

### Trusted Types API (W3C, 2025)

The W3C Trusted Types specification prevents DOM XSS by enforcing that all dangerous sinks (innerHTML, document.write, eval) only accept typed objects, not raw strings. Google's adoption study found Trusted Types eliminated DOM XSS in production web frameworks.

**Source:** https://web.dev/articles/trusted-types
**Spec:** https://w3c.github.io/trusted-types/dist/spec/
**Research:** Kern, "Adopting Trusted Types in Production Web Frameworks", Google Research 2025. https://research.google/pubs/adopting-trusted-types-in-production-web-frameworks-to-prevent-dom-based-cross-site-scripting-a-case-study/

### Content Security Policy (OWASP + MDN, 2025)

A strict CSP should use nonce-based script sources rather than 'unsafe-inline'. The OWASP CSP Cheat Sheet recommends: `script-src 'nonce-{random}'; style-src 'nonce-{random}'; default-src 'self'; img-src 'self' data:; connect-src 'self' wss:`. The `require-trusted-types-for 'script'` directive adds an additional enforcement layer.

**Source:** https://cheatsheetseries.owasp.org/cheatsheets/Content_Security_Policy_Cheat_Sheet.html
**Source:** https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/CSP

### Token Storage (OWASP Session Management, 2025)

OWASP and multiple security analyses recommend: store short-lived access tokens in memory (JS variable), store refresh tokens in HttpOnly SameSite=Strict cookies. localStorage is accessible to XSS payloads and should never hold authentication tokens in security-sensitive applications.

**Source:** https://dev.to/cotter/localstorage-vs-cookies-all-you-need-to-know-about-storing-jwt-tokens-securely-in-the-front-end-15id
**Source:** https://www.wisp.blog/blog/understanding-token-storage-local-storage-vs-httponly-cookies

## Proposed Implementation

### Phase 1: Fix Unescaped innerHTML Injections (agent_zero.html)

**1a. Escape `thought` in agent thought rendering (line 2740):**
```javascript
// BEFORE (line 2740):
headerHtml += `<span class="agent-thought-text">${thought}</span>`;

// AFTER:
headerHtml += `<span class="agent-thought-text">${esc(thought)}</span>`;
```

**1b. Escape `agentName` (line 2739):**
```javascript
// BEFORE:
headerHtml += `<span class="agent-thought-name">${agentName}</span>`;

// AFTER:
headerHtml += `<span class="agent-thought-name">${esc(agentName)}</span>`;
```

**1c. Escape `req.category` and `req.priority` in renderRequest (line 3274):**
```javascript
// BEFORE:
<div class="req-meta">${req.category || 'feature'} | ${req.priority || 'medium'} | ...

// AFTER:
<div class="req-meta">${esc(req.category || 'feature')} | ${esc(req.priority || 'medium')} | ...
```

**1d. Escape IDs in onclick handlers (lines 3278, 3459-3464, 3582-3590):**
```javascript
// BEFORE:
onclick="approveRequest('${req.id}')"

// AFTER:
onclick="approveRequest('${esc(req.id)}')"
```

Apply same pattern to all `g.id` and `c.id` onclick injections.

### Phase 2: Add DOMPurify for renderMarkdown Output

**2a. Load DOMPurify (add before closing `</head>`):**
```html
<script src="https://cdnjs.cloudflare.com/ajax/libs/dompurify/3.2.4/purify.min.js"
        integrity="sha384-..." crossorigin="anonymous"></script>
```

**2b. Wrap renderMarkdown output (lines 2574, 2624, 3377):**
```javascript
// BEFORE:
contentDiv.innerHTML = renderMarkdown(content);

// AFTER:
contentDiv.innerHTML = DOMPurify.sanitize(renderMarkdown(content));
```

### Phase 3: Add Content Security Policy Header (agent_zero_server.py)

**3a. Generate nonce per request and inject into HTML template:**

In `agent_zero_server.py`, add to the HTML serving endpoint:
```python
import secrets

@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    nonce = secrets.token_urlsafe(16)
    html = templates.TemplateResponse("agent_zero.html", {"request": request, "nonce": nonce})
    html.headers["Content-Security-Policy"] = (
        f"default-src 'self'; "
        f"script-src 'nonce-{nonce}' https://cdnjs.cloudflare.com; "
        f"style-src 'nonce-{nonce}'; "
        f"connect-src 'self' wss: ws:; "
        f"img-src 'self' data:; "
        f"font-src 'self'; "
        f"object-src 'none'; "
        f"base-uri 'self'; "
        f"form-action 'self'"
    )
    return html
```

**3b. Add nonce to inline script and style tags in agent_zero.html:**
```html
<style nonce="{{ nonce }}">...</style>
<script nonce="{{ nonce }}">...</script>
```

### Phase 4: Fix Empty Catch Blocks

**4a. Add error logging to silent catch blocks (lines 3293, 3308, 3445):**
```javascript
// BEFORE:
} catch {}

// AFTER:
} catch (err) { console.error('Request failed:', err); }
```

### Phase 5: Tighten CORS (agent_zero_server.py)

**5a. Replace wildcard allow_headers (line 187):**
```python
# BEFORE:
allow_headers=["*"],

# AFTER:
allow_headers=["Authorization", "Content-Type"],
```

## Test Specifications

### XSS Prevention Tests

```python
def test_agent_thought_xss_escaped():
    """Verify agent thought content is HTML-escaped before display."""
    # Send WebSocket message with thought containing <script>alert(1)</script>
    # Verify rendered HTML contains &lt;script&gt; not <script>

def test_request_category_xss_escaped():
    """Verify request category/priority fields are escaped."""
    # Create request with category="<img onerror=alert(1)>"
    # Verify rendered HTML escapes the img tag

def test_onclick_id_injection_prevented():
    """Verify IDs in onclick handlers are escaped."""
    # Create goal with id="'); alert(1); //"
    # Verify onclick contains escaped quotes

def test_rendermarkdown_code_block_safe():
    """Verify code blocks don't execute scripts."""
    # Input: ```\n<script>alert(1)</script>\n```
    # Verify output contains &lt;script&gt; inside <code>

def test_dompurify_strips_script_tags():
    """Verify DOMPurify removes script tags from markdown output."""
    # Input markdown with embedded script after processing
    # Verify DOMPurify.sanitize removes it
```

### CSP Tests

```python
def test_csp_header_present():
    """Verify Content-Security-Policy header on HTML response."""
    response = client.get("/")
    assert "Content-Security-Policy" in response.headers
    csp = response.headers["Content-Security-Policy"]
    assert "script-src" in csp
    assert "'unsafe-inline'" not in csp

def test_csp_nonce_rotates():
    """Verify CSP nonce changes per request."""
    r1 = client.get("/")
    r2 = client.get("/")
    nonce1 = extract_nonce(r1.headers["Content-Security-Policy"])
    nonce2 = extract_nonce(r2.headers["Content-Security-Policy"])
    assert nonce1 != nonce2

def test_cors_headers_not_wildcard():
    """Verify CORS allow_headers is not wildcard."""
    # Send OPTIONS request
    # Verify Access-Control-Allow-Headers is specific list, not *
```

### Error Handling Tests

```python
def test_approve_request_error_logged():
    """Verify approveRequest error is not silently swallowed."""
    # Mock fetch to fail
    # Verify console.error was called (or UI shows error state)
```

## Estimated Impact

- **Security:** Eliminates DOM XSS attack surface (OWASP A03:2021). XSS is the #3 most critical web vulnerability.
- **Defense-in-depth:** CSP header prevents exploitation even if individual escaping is missed.
- **Token safety:** Moving JWT from localStorage to HttpOnly cookies prevents token theft via XSS (deferred to separate paper since it requires backend auth flow changes).
- **Compliance:** Meets OWASP ASVS Level 2 requirements for output encoding (V5.3) and CSP (V14.4).
- **Risk reduction:** The `thought` field comes directly from LLM output -- if the model is prompt-injected, the current code would execute arbitrary HTML/JS in the user's browser.
