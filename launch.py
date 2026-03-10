"""
AgentZero launcher.
Runs sessions back-to-back forever.
Ctrl+C or drop a STOP file in Z:\AgentZero\ to halt.
"""
import subprocess, os, time, datetime, signal, threading, json

WORKSPACE    = "Z:/AgentZero"
STOP_FILE    = os.path.join(WORKSPACE, "STOP")
RESTART_WAIT = 3

halted = threading.Event()

def on_sigint(sig, frame):
    print("\n[ctrl+c] Writing STOP — halting after this session...")
    open(STOP_FILE, "w").write(f"stopped at {datetime.datetime.now().isoformat()}\n")
    halted.set()

signal.signal(signal.SIGINT, on_sigint)

def run_session(cycle):
    if os.path.exists(STOP_FILE):
        os.remove(STOP_FILE)

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n=== Session {cycle} started — {ts} ===\n")

    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("CLAUDE")}

    proc = subprocess.Popen(
        ["claude", "--model", "claude-opus-4-6",
         "--effort", "medium",
         "--dangerously-skip-permissions",
         "--output-format", "stream-json", "--verbose",
         "-p", "Read your CLAUDE.md and begin."],
        cwd=WORKSPACE, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )

    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            t = d.get("type", "")
            if t == "assistant":
                for blk in d.get("message", {}).get("content", []):
                    if blk.get("type") == "text" and blk["text"].strip():
                        print(f"\n[agent] {blk['text'].strip()}\n")
                    elif blk.get("type") == "tool_use":
                        name = blk.get("name", "?")
                        inp  = blk.get("input", {})
                        arg  = (inp.get("command") or inp.get("file_path")
                                or inp.get("pattern") or inp.get("query")
                                or str(inp)[:80])
                        print(f"[tool]  {name}({str(arg)[:120]})")
            elif t == "tool_result":
                content = d.get("content", "")
                snippet = ""
                if isinstance(content, list):
                    for c in content:
                        if c.get("type") == "text":
                            snippet = c["text"].strip()[:200].replace("\n", " ")
                            break
                elif isinstance(content, str):
                    snippet = content.strip()[:200]
                if snippet:
                    print(f"        -> {snippet}")
            elif t == "result" and d.get("is_error"):
                print(f"[error] {str(d.get('error', ''))[:200]}")
        except json.JSONDecodeError:
            if line:
                print(line)

    proc.wait()

cycle = 1
print("AgentZero — running forever. Ctrl+C or STOP file to halt.")

while not halted.is_set():
    run_session(cycle)
    cycle += 1

    if halted.is_set() or os.path.exists(STOP_FILE):
        break

    print(f"\n... restarting in {RESTART_WAIT}s ...\n")
    time.sleep(RESTART_WAIT)

print(f"\nAgentZero halted after {cycle - 1} session(s).")
