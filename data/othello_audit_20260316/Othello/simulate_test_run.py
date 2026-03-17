import time
import os
import json
import asyncio
from datetime import datetime, timezone
from core.llm_wrapper import LLMWrapper
from dotenv import load_dotenv

# Load environment
load_dotenv()
chosen_model = LLMWrapper().model

# Agentic Agents
from modules.agentic_agents.shadow_agent import ShadowAgent
from modules.agentic_agents.decision_vault_agent import DecisionVaultAgent
from modules.agentic_agents.prism_agent import PrismAgent
from modules.agentic_agents.architect_agent import ArchitectAgent
from modules.agentic_agents.psyche_agent import PsycheAgent
from modules.agentic_agents.goal_management_agent import GoalManagementAgent
from fello import Fello
from othello import Othello

# Sub-agents
from modules.agents.behavioral_agent import BehavioralAgent
from modules.agents.reflective_agent import ReflectiveAgent
from modules.agents.routine_tracker_agent import RoutineTrackerAgent
from modules.agents.impatience_detection_agent import ImpatienceDetectionAgent
from modules.agents.trait_agent import TraitAgent
from modules.agents.conversation_agent import ConversationAgent

# Directories
TESTS_DIR = "tests"
SIM_LOG_DIR = "sim_logs"
DEBUG_LOG_DIR = "debug_logs"
SUMMARY_LOG_DIR = "summary_logs"
os.makedirs(SIM_LOG_DIR, exist_ok=True)
os.makedirs(DEBUG_LOG_DIR, exist_ok=True)
os.makedirs(SUMMARY_LOG_DIR, exist_ok=True)

# Agents
shadow_agent = ShadowAgent()
decision_agent = DecisionVaultAgent()
prism_agent = PrismAgent(central_hub=None, agentic_hub=None)
architect_agent = ArchitectAgent(central_hub=None, agentic_hub=None, model=chosen_model)
psyche_agent = PsycheAgent(central_hub=None, agentic_hub=None)
goal_agent = GoalManagementAgent(central_hub=None, agentic_hub=None)
fello = Fello()
othello = Othello()

behavioral_agent = BehavioralAgent()
reflective_agent = ReflectiveAgent()
routine_tracker_agent = RoutineTrackerAgent()
impatience_agent = ImpatienceDetectionAgent()
trait_agent = TraitAgent()
conversation_agent = ConversationAgent()


async def run_test_simulation():
    test_files = [f for f in os.listdir(TESTS_DIR) if f.startswith("test_") and f.endswith(".json")]
    total = len(test_files)
    passed = 0
    failed = 0

    print(f"[Simulation] Starting simulation across {total} test documents...\n")

    for idx, filename in enumerate(test_files, start=1):
        try:
            with open(os.path.join(TESTS_DIR, filename), "r", encoding="utf-8") as f:
                input_data = json.load(f)

            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
            print(f"[{idx}/{total}] Running test from: {filename}")

            debug_lines = []

            parsed_input = conversation_agent.parse(input_data["text"])
            debug_lines.append(f"[ConversationAgent] Parsed → {parsed_input}")

            shadow_output = shadow_agent.get_refined_shadow()
            debug_lines.append(f"[ShadowAgent] Refined shadow → {shadow_output}")

            behavior_output = behavioral_agent.analyze_behavior(shadow_output, input_data["mood"])
            debug_lines.append(f"[BehavioralAgent] Analysis → {behavior_output}")

            reflection_result = reflective_agent.run_full_reflection(shadow_output)
            debug_lines.append(f"[ReflectiveAgent] Reflection → {reflection_result}")

            routine_snapshot = routine_tracker_agent.build_snapshot()
            debug_lines.append(f"[RoutineTracker] Snapshot → {routine_snapshot}")

            impatience_result = impatience_agent.detect_impatience(input_data["text"], input_data["mood"])
            debug_lines.append(f"[ImpatienceDetection] → {impatience_result}")

            trait_agent.analyze_text(reflection_result.get("summary_text", ""))
            debug_lines.append(f"[TraitAgent] Traits extracted")

            prism_output = prism_agent.analyze_prism_state(shadow_output, behavior_output)
            debug_lines.append(f"[PrismAgent] Delta → {prism_output}")

            decision_output = decision_agent.analyze_decisions()
            debug_lines.append(f"[DecisionVault] Decisions → {decision_output}")

            architect_output = await architect_agent.plan_and_execute(input_data["text"])
            debug_lines.append(f"[ArchitectAgent] Plan → {architect_output}")

            psyche_output = await psyche_agent.analyze_psyche(shadow_output, behavior_output)
            debug_lines.append(f"[PsycheAgent] Output → {psyche_output}")

            goal_agent.parse_and_store_goals(parsed_input.get("goals", []))
            goal_agent_output = goal_agent.list_goals()
            debug_lines.append(f"[GoalAgent] Goals → {goal_agent_output}")

            fello_output = await fello.deliberate()
            debug_lines.append(f"[Fello] Deliberation → {fello_output}")

            othello_output = othello.validate_action(fello_output)
            debug_lines.append(f"[Othello] Judgement → {othello_output}")

            result = {
                "file": filename,
                "input": input_data,
                "parsed_input": parsed_input,
                "shadow_output": shadow_output,
                "behavior_output": behavior_output,
                "reflection_result": reflection_result,
                "routine_snapshot": routine_snapshot,
                "impatience_result": impatience_result,
                "prism_output": prism_output,
                "decision_output": decision_output,
                "architect_output": {"plan": architect_output},
                "psyche_output": psyche_output,
                "goal_agent_output": goal_agent_output,
                "fello_output": fello_output,
                "othello_output": othello_output,
                "timestamp": timestamp
            }

            with open(os.path.join(SIM_LOG_DIR, f"sim_{timestamp}.json"), "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

            with open(os.path.join(DEBUG_LOG_DIR, f"sim_{timestamp}_debug.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(debug_lines))

            with open(os.path.join(SUMMARY_LOG_DIR, f"sim_{timestamp}_summary.txt"), "w", encoding="utf-8") as f:
                f.write(f"--- SUMMARY {timestamp} ---\n")
                f.write(f"File: {filename}\nMood: {input_data['mood']}\nText: {input_data['text']}\n")
                f.write(f"Reflection: {reflection_result.get('summary_text', 'N/A')}\n")
                f.write(f"Behavior Insight: {behavior_output.get('summary', 'N/A')}\n")
                f.write(f"Impatience: {impatience_result.get('impatience_level', 'N/A')}\n")
                f.write(f"Key Traits: {prism_output.get('key_traits', [])}\n")
                f.write(f"Final Action: {othello_output.get('action', 'N/A')}\n")

            print(f"[Simulation] ✅ Completed → {timestamp} | File: {filename}")
            passed += 1

        except Exception as e:
            print(f"[Simulation] ❌ Failed → {filename} | Error: {str(e)}")
            failed += 1

        time.sleep(2)

    print("\n[Simulation Complete]")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print("Check `debug_logs/` for detailed traces.")

if __name__ == "__main__":
    asyncio.run(run_test_simulation())
