---
topic: React Frontend Error Resilience and Type Safety
status: in_progress
priority: high
estimated_complexity: medium
researched_at: 2026-03-18T22:00:00Z
---

# React Frontend Error Resilience and Type Safety

## Problem Statement

The agent_zero-ui React frontend (12 components, ~2000 lines) has zero error boundaries, uses `props: any` for the main AppShell component, has only 1 test file across the entire frontend, and concentrates all application state (~40 useState hooks) in a single 1000+ line App.tsx god component. Any runtime error in any component crashes the entire application with no recovery path.

## Current State in Agent Zero

### No Error Boundaries

The React frontend has zero `ErrorBoundary` components anywhere. Every `.tsx` file was checked:

- `agent_zero-ui/src/App.tsx` (1000+ lines) -- root component, no error boundary
- `agent_zero-ui/src/components/AppShell.tsx` (693 lines) -- renders all tabs, no error boundary
- `agent_zero-ui/src/components/CommitmentsView.tsx` -- no error boundary
- `agent_zero-ui/src/components/CheckInsView.tsx` -- no error boundary
- `agent_zero-ui/src/components/AgentChipGrid.tsx` -- no error boundary
- `agent_zero-ui/src/components/RuntimeSettingsModal.tsx` -- no error boundary

If any component throws during render (e.g., a malformed shadow profile, a null reference in reasoning trace data, or a failed markdown render), the entire React tree unmounts and the user sees a white screen.

### Untyped AppShell Props

`agent_zero-ui/src/components/AppShell.tsx:26`:
```tsx
export function AppShell(props: any) {
```

The AppShell component accepts `props: any`, meaning:
- No compile-time checking of 50+ props passed from App.tsx
- No IDE autocomplete or refactoring support
- Missing props silently produce `undefined` instead of compilation errors
- TypeScript's value is completely negated for the largest UI component

App.tsx passes props correctly with proper types internally, but the AppShell boundary throws away all type information.

### Minimal Test Coverage

Only 1 test file exists: `agent_zero-ui/src/hooks/useSessionPersistedState.test.ts`

Zero component tests exist for:
- AppShell (693 lines, renders all 7 tabs)
- CommitmentsView (CRUD operations, form validation)
- CheckInsView (user interactions)
- AuthScreen (login/register flow)
- AgentChipGrid (reasoning step display)
- RuntimeSettingsModal (configuration changes)

### God Component Anti-Pattern

`agent_zero-ui/src/App.tsx` contains:
- ~40 `useState` hooks (lines 68-180)
- WebSocket connection management
- Audio recording/playback logic
- Authentication flows
- Desktop runtime management
- Onboarding state
- All data fetching (history, goals, commitments, requests, observations, reasoning)

This violates the Single Responsibility Principle and makes the component untestable in isolation.

### dangerouslySetInnerHTML Usage

`AppShell.tsx:161` and `AppShell.tsx:348`:
```tsx
{ dangerouslySetInnerHTML: { __html: renderMarkdown(String(message.content ?? '')) } }
```

The `renderMarkdown` function (`lib/format.ts:69-98`) does HTML-escape `<`, `>`, `&` before adding structural HTML, which is safe. However, the regex-based markdown parser is fragile -- edge cases in the model tag stripping (lines 72-77) could allow injection if patterns don't match exactly. A dedicated sanitization library would eliminate this risk class entirely.

## Industry Standard / Research Findings

### Error Boundaries (React 19)

React's official documentation and the broader ecosystem strongly recommend strategic error boundary placement around independent feature areas rather than a single top-level boundary (React Docs, https://react.dev/reference/react/Component#catching-rendering-errors-with-an-error-boundary).

OneUptime (2026) documents the "write once, use everywhere" pattern: a single reusable `ErrorBoundary` class component that accepts a `fallback` prop and optional `onError` callback for telemetry. This prevents cascading failures by isolating crashes to the specific panel that failed.
URL: https://oneuptime.com/blog/post/2026-02-20-react-error-boundaries/view

React 19 improves error boundary behavior with better error reporting and no duplicate console logs (Hash Block, 2025, https://medium.com/@connect.hashblock/react-19-resilience-retry-suspense-error-boundaries-40ea504b09ed).

The `react-error-boundary` library by Brian Vaughn (now part of the React ecosystem) provides a declarative API with `useErrorBoundary` hook for imperative error throwing and `ErrorBoundary` with reset keys, reducing boilerplate.
URL: https://github.com/bvaughn/react-error-boundary

### TypeScript Type Safety for React

Perficient (2025) and the TypeScript React Cheatsheets project both document that `props: any` defeats TypeScript's purpose. The standard is to define explicit interfaces for all component props, enabling compile-time verification of prop contracts between parent and child components.
URL: https://blogs.perficient.com/2025/03/05/using-typescript-with-react-best-practices/
URL: https://github.com/typescript-cheatsheets/react

SitePoint (2025) recommends using `ComponentProps<typeof Component>` utility types when wrapping components to maintain type inheritance chains.
URL: https://www.sitepoint.com/react-with-typescript-best-practices/

### Component Decomposition

Leapcell (2025) documents practical strategies for decomposing large components: extract custom hooks for state logic, split rendering into focused sub-components, and use composition patterns to avoid prop drilling.
URL: https://leapcell.io/blog/practical-strategies-for-decomposing-large-components-in-react-and-vue

Frontend Mastery documents the "lift content up, push state down" pattern where global state consumption should be pushed as close as possible to the components that render UI based on that state.
URL: https://frontendmastery.com/posts/advanced-react-component-composition-guide/

### Frontend Testing

DEV Community (2025) and Kent C. Dodds' Testing Library documentation recommend testing from the user's perspective using `getByRole` as the default query. Focus on behavior, not implementation details.
URL: https://dev.to/tahamjp/react-component-testing-best-practices-for-2025-2674
URL: https://kentcdodds.com/blog/common-mistakes-with-react-testing-library

The recommendation is 70-80% meaningful coverage focused on business-critical flows before edge cases.

## Proposed Implementation

### Phase 1: Error Boundary Component (new file)

Create `agent_zero-ui/src/components/ErrorBoundary.tsx`:

```tsx
import { Component, type ReactNode } from 'react'

interface Props {
  children: ReactNode
  fallback?: ReactNode
  featureName?: string
  onError?: (error: Error, errorInfo: { componentStack: string }) => void
}

interface State {
  hasError: boolean
  error: Error | null
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, error: null }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error(`[ErrorBoundary:${this.props.featureName ?? 'unknown'}]`, error, errorInfo)
    this.props.onError?.(error, { componentStack: errorInfo.componentStack ?? '' })
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback ?? (
        <div className="error-fallback">
          <p>Something went wrong in {this.props.featureName ?? 'this section'}.</p>
          <button type="button" onClick={() => this.setState({ hasError: false, error: null })}>
            Try again
          </button>
        </div>
      )
    }
    return this.props.children
  }
}
```

### Phase 2: Wrap Independent Feature Areas in AppShell.tsx

Each tab panel and each modal should be wrapped in its own ErrorBoundary:

```tsx
// Chat panel
<ErrorBoundary featureName="Chat">
  <section className="panel chat-panel">...</section>
</ErrorBoundary>

// History panel
<ErrorBoundary featureName="History">
  <section className="panel">...</section>
</ErrorBoundary>

// Runtime panel, Shadow panel, Trace panel -- each wrapped individually
// Modals: RuntimeSettingsModal, AgentDetailModal, onboarding overlay -- each wrapped
```

This ensures a crash in the Shadow tab doesn't take down the Chat tab.

### Phase 3: Type AppShell Props

Replace `props: any` in AppShell.tsx with a proper interface:

```tsx
interface AppShellProps {
  // Auth
  currentUser: User
  authToken: string
  onLogout: () => void

  // Navigation
  activeTab: Tab
  onActiveTabChange: (tab: Tab) => void

  // Chat
  messages: ChatMessage[]
  messageInput: string
  onMessageInputChange: (value: string) => void
  onSendMessage: (e: FormEvent) => void
  isGenerating: boolean

  // Reasoning
  reasoningSteps: ReasoningStep[]
  chipsByTurn: Record<number, ReasoningStep[]>
  memoryContextByTurn: Record<number, MemoryContext>
  toolActivityByTurn: Record<number, ToolActivity>

  // Voice
  speechPlaybackEnabled: boolean
  onSpeechPlaybackToggle: () => void
  // ... (complete all ~50 props)
}

export function AppShell(props: AppShellProps) {
```

### Phase 4: Extract Custom Hooks from App.tsx

Extract the largest state clusters into custom hooks:

1. **`useWebSocket(token)`** -- WebSocket connection, reconnection, message dispatch (~150 lines)
2. **`useAuth()`** -- login, register, logout, token management (~100 lines)
3. **`useVoice()`** -- recording, playback, wave bars, speech synthesis (~120 lines)
4. **`useDataFetching(token, userId)`** -- goals, commitments, history, requests, observations (~200 lines)

This reduces App.tsx from ~1000 lines to ~300 lines of composition logic.

### Phase 5: Add DOMPurify Sanitization

Add `dompurify` package and update `renderMarkdown`:

```tsx
import DOMPurify from 'dompurify'

export function renderMarkdown(text: string) {
  // ... existing markdown conversion ...
  return DOMPurify.sanitize(html)
}
```

This eliminates the entire XSS risk class for `dangerouslySetInnerHTML` usage.

### Phase 6: Component Tests

Add test files using Vitest + React Testing Library:

1. `AppShell.test.tsx` -- renders each tab, displays messages, handles user input
2. `CommitmentsView.test.tsx` -- CRUD flows, form validation
3. `AuthScreen.test.tsx` -- login/register modes, error display
4. `ErrorBoundary.test.tsx` -- catches errors, shows fallback, reset works

Target: 70%+ coverage of user-facing behavior.

## Test Specifications

### ErrorBoundary Tests

```python
# test_error_boundary (conceptual -- actual tests in TypeScript/Vitest)

def test_error_boundary_renders_children_when_no_error():
    """ErrorBoundary renders children normally when no error occurs."""
    # Render <ErrorBoundary><ChildComponent /></ErrorBoundary>
    # Assert child content is visible

def test_error_boundary_shows_fallback_on_child_crash():
    """ErrorBoundary shows fallback UI when child throws during render."""
    # Render <ErrorBoundary featureName="Test"><CrashingComponent /></ErrorBoundary>
    # Assert "Something went wrong in Test" is visible
    # Assert child content is NOT visible

def test_error_boundary_reset_recovers():
    """Clicking 'Try again' resets the error boundary and re-renders children."""
    # After crash, click reset button
    # Assert children render again (if underlying error is fixed)

def test_error_boundary_calls_onError_callback():
    """ErrorBoundary calls onError prop with error details."""
    # Render with onError mock
    # Trigger child crash
    # Assert onError was called with Error and componentStack

def test_error_boundary_isolates_panel_crashes():
    """A crash in one panel does not affect other panels."""
    # Render two ErrorBoundary-wrapped panels
    # Crash one panel
    # Assert the other panel still renders correctly
```

### Type Safety Tests

```python
def test_appshell_props_type_check():
    """TypeScript compilation succeeds with correct props and fails with missing required props."""
    # This is verified by tsc --noEmit (TypeScript compiler check)
    # Add to CI: npx tsc --noEmit

def test_appshell_renders_with_typed_props():
    """AppShell renders correctly when given properly typed props."""
    # Create mock props conforming to AppShellProps interface
    # Render <AppShell {...mockProps} />
    # Assert no runtime errors
```

### Sanitization Tests

```python
def test_render_markdown_sanitizes_script_tags():
    """renderMarkdown strips <script> tags even if they bypass regex stripping."""
    # Input: "Hello <script>alert(1)</script> world"
    # Output should NOT contain <script>

def test_render_markdown_sanitizes_event_handlers():
    """renderMarkdown strips onerror/onload attributes."""
    # Input: '<img src=x onerror=alert(1)>'
    # Output should NOT contain onerror
```

## Estimated Impact

- **Crash resilience**: Individual panel failures no longer crash the entire app. Users can continue using unaffected tabs.
- **Developer velocity**: Typed props catch ~30% of integration bugs at compile time (TypeScript team estimates).
- **Maintainability**: Custom hooks reduce App.tsx cognitive complexity from O(n) state variables to O(k) composable units.
- **Security**: DOMPurify eliminates an entire vulnerability class regardless of future markdown parser changes.
- **Testability**: Component tests catch regressions before deployment, especially for CRUD-heavy flows like commitments and check-ins.
