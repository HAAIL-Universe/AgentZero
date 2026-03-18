---
topic: Frontend Accessibility (WCAG 2.1 AA)
status: ready_for_implementation
priority: medium
estimated_complexity: large
researched_at: 2026-03-18T00:00:00Z
---

# Frontend Accessibility (WCAG 2.1 AA)

## Problem Statement

agent_zero.html (4,101 lines) has **zero ARIA attributes**, **zero landmark roles**, **zero live regions**, and **zero skip-navigation links**. The application is a real-time chat and voice interface -- precisely the type of UI that requires the most accessibility work. The current state is estimated below WCAG 2.1 Level A compliance. This blocks deployment to any context where ADA Title II applies (US public entities, April 2026 deadline) and excludes screen reader users entirely.

## Current State in Agent Zero

**File:** `agent_zero/templates/agent_zero.html` (~4,101 lines)

### What exists:
- **3 `<label>` elements** -- auth form only (lines 1673, 1678, 1683)
- **35 `<button>` elements** -- most have text content (accessible names present)
- **3 keyboard handlers** -- Enter-to-send on message input (line 4072), auth password (line 4085), goal input (line 4090)
- **1 programmatic focus()** -- message input after speech playback (line 2691)
- **1 autofocus** -- message-input textarea (line 1803)
- **CSS focus styles** -- auth fields (line 161), message input (line 543), onboarding inputs (lines 1154-1155)

### What is missing:
| Category | Count | Status |
|----------|-------|--------|
| ARIA attributes (any) | 0 | Critical gap |
| Role attributes | 0 | Critical gap |
| Tabindex attributes | 0 | Gap |
| Live regions (aria-live) | 0 | Critical -- chat messages invisible to screen readers |
| Skip-nav links | 0 | Missing |
| Semantic landmarks (header/nav/main) | 0 | All `<div>` |
| Tab panel semantics | 0 | 9 tabs lack role="tablist"/role="tab" |
| Modal accessibility | 0 | 2 overlays lack role="dialog", focus trap, Escape |
| Form labels (non-auth) | 0 | 6 inputs rely on placeholder only |
| Toggle button states (aria-pressed) | 0 | Voice on/off button unlabeled |

### Specific elements needing remediation:

1. **Chat message container** (`#messages`, ~line 1800) -- no role="log", no aria-live
2. **Tab bar** (lines 1700-1711) -- 9 buttons, no role="tablist"/"tab"/"tabpanel", no aria-selected/aria-controls
3. **Header** (`#header`, line 1657) -- `<div>` not `<header>`
4. **Toolbar** (`#toolbar`, line 1713) -- `<div>` not role="toolbar"
5. **Voice overlay** (`#voice-overlay`, line 1765) -- no role="dialog", no aria-modal, no focus trap, no Escape handler
6. **Onboarding overlay** (`#onboarding-overlay`, line 1908) -- same issues as voice overlay
7. **Message input** (line 1803) -- `<textarea>` with placeholder only, no `<label>`
8. **Goal input** (line 1828) -- placeholder only
9. **Commitment inputs** (lines 1859-1865) -- 3 inputs with no labels
10. **Auth error** (`#auth-error`, line 1689) -- no role="alert" or aria-live
11. **Status dot** (`.dot-ready`/`.dot-loading`/`.dot-offline`) -- no text alternative
12. **Color contrast** -- `--text-dim: #555570` (4.07:1 on #1a1a2e bg, fails AA for normal text), `.dot-offline: #666` (barely visible)

## Industry Standard / Research Findings

### WCAG 2.1 AA Success Criteria Applicable to Chat/Voice Interfaces

| SC | Name | Relevance | Agent Zero Status |
|----|------|-----------|----------------|
| 1.3.1 | Info and Relationships | Chat structure must be programmatically determinable | FAIL -- no semantic markup |
| 1.4.3 | Contrast (Minimum) | 4.5:1 for normal text, 3:1 for large text | FAIL -- `--text-dim` is 4.07:1 |
| 2.1.1 | Keyboard | All functions operable by keyboard | PARTIAL -- Send works, overlays don't |
| 2.4.1 | Bypass Blocks | Skip-nav links required | FAIL -- none present |
| 2.4.3 | Focus Order | Logical focus order | FAIL -- overlays don't trap focus |
| 2.4.7 | Focus Visible | Clear focus indicators | PARTIAL -- some CSS focus styles |
| 2.5.3 | Label in Name | Visible label matches accessible name | FAIL -- many inputs unlabeled |
| 4.1.3 | Status Messages | Programmatic status for screen readers | FAIL -- no live regions |

### W3C ARIA Authoring Practices Patterns for Chat

**role="log" pattern** (ARIA23 technique) -- the canonical accessible chat structure:

```html
<div id="chatRegion" role="log" aria-labelledby="chatHeading" aria-live="polite">
  <h4 id="chatHeading" class="sr-only">Chat History</h4>
  <ul id="conversation">
    <li>The latest chat message</li>
  </ul>
</div>
```

Key defaults for role="log": implicit aria-live="polite", aria-atomic="false", aria-relevant="additions". MDN recommends adding explicit aria-live="polite" for broader screen reader compatibility.

**Dual-region pattern for WebSocket chat** (TetraLogical, 2024): Use role="log" for the visual message history, plus a separate hidden live region for screen reader announcements. This prevents entire chat history re-read on each new message.

```html
<!-- Visual chat log (scrollable) -->
<div id="messages" role="log" aria-labelledby="chat-heading">...</div>

<!-- Hidden announcer for screen readers -->
<div id="sr-announcer" aria-live="polite" class="sr-only"></div>
```

**Critical rule**: Live region containers MUST exist in DOM before content injection. Pre-populated regions are not announced (W3C ARIA Working Group confirmed this as intended behavior).

**Feed pattern** (APG) for scrollback history: role="feed" with aria-posinset/aria-setsize on message articles. Keyboard: Page Down/Up for article navigation, Ctrl+End/Home to exit feed.

**Toggle button pattern** for record button: aria-pressed="false"/"true" with constant label (screen readers announce "Record, toggle button, pressed" automatically).

### References

1. **W3C Understanding SC 4.1.3 (Status Messages)** -- https://www.w3.org/WAI/WCAG21/Understanding/status-messages.html -- Defines ARIA22 (role="status"), ARIA23 (role="log"), ARIA19 (role="alert") techniques for programmatic status.

2. **TetraLogical: Why Are My Live Regions Not Working?** (May 2024) -- https://tetralogical.com/blog/2024/05/01/why-are-my-live-regions-not-working/ -- Documents pre-population bug across browser/SR combinations. Firefox+JAWS announces pre-populated content (spec violation). Recommends setTimeout between container creation and content injection.

3. **W3C ARIA Authoring Practices Guide: Feed Pattern** -- https://www.w3.org/WAI/ARIA/apg/patterns/feed/ -- Defines keyboard model for infinite-scroll content. role="feed" container, role="article" items with aria-posinset/aria-setsize.

4. **Sara Soueidan: Accessible Notifications with ARIA Live Regions** (2024) -- https://www.sarasoueidan.com/blog/accessible-notifications-with-aria-live-regions-part-1/ -- Comprehensive guide to live region pitfalls. Recommends never combining role="alert" with aria-live="assertive" (iOS VoiceOver double-speaks).

5. **WCAG 2.2 ISO/IEC 40500:2025** (Oct 2025) -- https://adaquickscan.com/blog/wcag-2-2-iso-standard-2025 -- WCAG 2.2 became ISO standard, adding SC 2.5.8 (Target Size, 24x24px min). US ADA Title II requires WCAG 2.1 AA by April 2026.

6. **CHI 2025: Designing Accessible Audio Nudges for Voice Interfaces** -- https://dl.acm.org/doi/10.1145/3706598.3713563 -- Design patterns for audio cues in voice-first interfaces for users with disabilities. Proposes multi-modal feedback beyond visual.

7. **MDN: ARIA Live Regions** -- https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/Guides/Live_regions -- Reference for aria-live, aria-atomic, aria-relevant attributes. Notes throttling needed for high-frequency updates.

## Proposed Implementation

Implementation is divided into 4 phases, each independently deployable.

### Phase 1: Semantic Landmarks and Skip Navigation (small, no JS changes)

**File:** `agent_zero/templates/agent_zero.html`

1. **Add skip-nav link** at top of `<body>`:
   ```html
   <a href="#messages" class="sr-only sr-only-focusable">Skip to chat</a>
   ```

2. **Add sr-only CSS class** (visually hidden, screen-reader visible):
   ```css
   .sr-only {
     position: absolute; width: 1px; height: 1px;
     padding: 0; margin: -1px; overflow: hidden;
     clip: rect(0,0,0,0); white-space: nowrap; border: 0;
   }
   .sr-only-focusable:focus {
     position: static; width: auto; height: auto;
     overflow: visible; clip: auto; white-space: normal;
   }
   ```

3. **Replace `<div id="header">` with `<header id="header">`** (line 1657)

4. **Add role="main"** to `<div id="chat-screen">` (line 1698)

5. **Add role="toolbar"** to `<div id="toolbar">` (line 1713)

### Phase 2: Tab Panel Semantics (small, minor JS changes)

**File:** `agent_zero/templates/agent_zero.html`

1. **Wrap tab buttons** in `role="tablist"`:
   ```html
   <div id="tab-bar" role="tablist" aria-label="Main navigation">
   ```

2. **Add tab attributes** to each tab button:
   ```html
   <button class="tab-btn" role="tab" aria-selected="true" aria-controls="chat-panel" id="tab-chat">Chat</button>
   <button class="tab-btn" role="tab" aria-selected="false" aria-controls="voice-panel" id="tab-voice">Voice</button>
   <!-- ... for all 9 tabs -->
   ```

3. **Add tabpanel role** to each panel:
   ```html
   <div id="chat-panel" role="tabpanel" aria-labelledby="tab-chat">
   ```

4. **Update tab-switching JS** (search for tab switching logic, likely in `showTab()` or similar function):
   - Set `aria-selected="true"` on active tab, `"false"` on others
   - Set `aria-hidden="true"` on inactive panels (in addition to display:none)

5. **Add keyboard navigation** to tab bar:
   - Arrow Left/Right to move between tabs
   - Home/End to jump to first/last tab
   - Tab key moves focus into the panel content

### Phase 3: Live Regions and Form Labels (medium, JS changes for dual-region pattern)

**File:** `agent_zero/templates/agent_zero.html`

1. **Chat messages live region** -- add to `#messages` container:
   ```html
   <div id="messages" role="log" aria-labelledby="chat-heading" aria-live="polite">
     <h2 id="chat-heading" class="sr-only">Chat messages</h2>
   </div>
   ```

2. **Add hidden announcer** for screen readers (separate from visual log):
   ```html
   <div id="sr-announcer" aria-live="polite" class="sr-only"></div>
   ```

3. **Update message injection JS** -- when a new Agent Zero message completes streaming, copy its text content to `#sr-announcer`:
   ```javascript
   // After message streaming completes:
   const announcer = document.getElementById('sr-announcer');
   announcer.textContent = ''; // Clear first
   setTimeout(() => {
     announcer.textContent = 'Agent Zero: ' + messageText;
   }, 100); // Allow SR to detect the change
   ```

4. **Auth error alert** -- add to `#auth-error` (line 1689):
   ```html
   <div id="auth-error" role="alert" aria-live="assertive" style="display:none"></div>
   ```

5. **Status dot** -- add aria-label to connection indicator:
   ```html
   <span class="status-dot dot-ready" aria-label="Connected"></span>
   ```
   Update dynamically when status changes.

6. **Form labels** -- add `<label>` elements or aria-label to unlabeled inputs:
   ```html
   <label for="message-input" class="sr-only">Type your message</label>
   <label for="goal-input" class="sr-only">New goal</label>
   <label for="commitment-input" class="sr-only">New commitment</label>
   <label for="commitment-cadence" class="sr-only">Frequency</label>
   <label for="commitment-weight" class="sr-only">Importance</label>
   ```

7. **Voice button** -- add aria-pressed toggle:
   ```html
   <button id="voice-toggle" aria-pressed="false" aria-label="Voice recording">
   ```
   Update `aria-pressed` in JS when recording starts/stops.

### Phase 4: Modal Accessibility (medium, JS changes for focus trap)

**File:** `agent_zero/templates/agent_zero.html`

1. **Voice overlay** -- add dialog semantics:
   ```html
   <div id="voice-overlay" role="dialog" aria-modal="true" aria-label="Voice recording">
   ```

2. **Onboarding overlay** -- add dialog semantics:
   ```html
   <div id="onboarding-overlay" role="dialog" aria-modal="true" aria-label="Orientation questions">
   ```

3. **Focus trap** -- when overlay opens:
   ```javascript
   function trapFocus(overlayEl) {
     const focusable = overlayEl.querySelectorAll(
       'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
     );
     const first = focusable[0];
     const last = focusable[focusable.length - 1];
     first.focus();

     overlayEl.addEventListener('keydown', (e) => {
       if (e.key === 'Escape') {
         closeOverlay(overlayEl);
         return;
       }
       if (e.key === 'Tab') {
         if (e.shiftKey && document.activeElement === first) {
           e.preventDefault();
           last.focus();
         } else if (!e.shiftKey && document.activeElement === last) {
           e.preventDefault();
           first.focus();
         }
       }
     });
   }
   ```

4. **Restore focus** -- when overlay closes, return focus to the element that triggered it.

### Color Contrast Fix (inline with any phase)

1. **`--text-dim: #555570`** (4.07:1 on #1a1a2e) -- change to `#6b6b8a` (4.58:1, passes AA)
2. **`.dot-offline: #666`** -- change to `#888` (visible) or add text label

## Test Specifications

### Phase 1 Tests (Semantic Landmarks)
```python
def test_skip_nav_link_exists():
    """agent_zero.html contains a skip-nav link targeting #messages"""

def test_header_element_is_semantic():
    """<header> tag used instead of <div id='header'>"""

def test_main_landmark_exists():
    """role='main' or <main> element present"""

def test_sr_only_class_exists():
    """CSS class sr-only defined with standard clip-rect pattern"""
```

### Phase 2 Tests (Tab Panel)
```python
def test_tablist_role():
    """Tab container has role='tablist'"""

def test_tab_roles():
    """Each tab button has role='tab'"""

def test_aria_selected_on_active_tab():
    """Active tab has aria-selected='true', others 'false'"""

def test_aria_controls_links_to_panel():
    """Each tab's aria-controls matches a tabpanel id"""

def test_tabpanel_roles():
    """Each content panel has role='tabpanel'"""

def test_tab_keyboard_navigation():
    """Arrow keys cycle through tabs, Home/End jump to first/last"""
```

### Phase 3 Tests (Live Regions)
```python
def test_messages_has_role_log():
    """#messages has role='log'"""

def test_messages_has_aria_live():
    """#messages has aria-live='polite'"""

def test_sr_announcer_exists():
    """Hidden #sr-announcer with aria-live='polite' exists"""

def test_auth_error_has_role_alert():
    """#auth-error has role='alert'"""

def test_all_inputs_have_labels():
    """Every input/textarea/select has an associated <label> or aria-label"""

def test_voice_button_has_aria_pressed():
    """Voice toggle button has aria-pressed attribute"""

def test_status_dot_has_label():
    """Connection status indicator has aria-label"""
```

### Phase 4 Tests (Modals)
```python
def test_voice_overlay_is_dialog():
    """#voice-overlay has role='dialog' and aria-modal='true'"""

def test_onboarding_overlay_is_dialog():
    """#onboarding-overlay has role='dialog' and aria-modal='true'"""

def test_overlay_escape_closes():
    """Pressing Escape while overlay is open closes it"""

def test_overlay_focus_trap():
    """Tab cycling within overlay wraps from last to first focusable element"""

def test_overlay_restores_focus():
    """Closing overlay returns focus to trigger element"""
```

### Color Contrast Tests
```python
def test_text_dim_contrast_ratio():
    """--text-dim color has >= 4.5:1 contrast ratio against --bg-primary"""

def test_dot_offline_visible():
    """Offline indicator has text alternative or sufficient contrast"""
```

## Estimated Impact

- **Screen reader users**: Currently 100% excluded from using Agent Zero. After implementation, full navigation and message reading possible.
- **Keyboard-only users**: Currently blocked by modal overlays with no Escape/Tab handling. After implementation, full keyboard operability.
- **WCAG compliance**: Moves from below Level A to Level AA compliance.
- **Legal exposure**: Mitigates ADA Title II risk (April 2026 deadline for WCAG 2.1 AA).
- **User base**: ~15% of world population has some disability (WHO). ~8% use assistive technology on the web.
- **SEO/structure**: Semantic landmarks improve search engine understanding.

Implementation order: Phase 1 (landmarks) and Phase 3 (live regions) provide the highest accessibility ROI. Phase 2 (tabs) and Phase 4 (modals) complete the picture. Each phase is independently deployable and testable.
