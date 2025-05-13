# TASK NAME
Update Project Documentation and Scripts for Current Models

## SUMMARY
Revise `README.md` and the `install_model` command in `autonomous_agent/main.py` to accurately reflect the current primary coding model (`deepseek-coder:6.7b-instruct`), the planned reasoning model (`phi3:mini`), and the dual-model architecture concept. Update `dev_log/summary.md` to match.

## REQUIREMENTS
1.  `README.md` must correctly state `deepseek-coder:6.7b-instruct` as the primary model, mention `phi3:mini` as the reasoning model in the dual-model setup, and update installation/usage instructions accordingly.
2.  The `install_model` command in `autonomous_agent/main.py` must be updated to pull `deepseek-coder:6.7b-instruct` (or a suitable default related to it) and `phi3:mini`.
3.  `dev_log/summary.md` must be updated to remove outdated references to `Wizard-Vicuna-13B-Uncensored` and reflect the current model strategy.
4.  Information about `Wizard-Vicuna-13B-Uncensored` should be removed or corrected to reflect its deprecated status across these files.

## FILE TREE:
*   `README.md` (Target file for update, located in project root)
*   `autonomous_agent/main.py` (Target file for `install_model` command update)
*   `dev_log/summary.md` (Target file for model information update)
*   `dev_log/model_change.md` (Source for DeepSeek-Coder info)
*   `dev_log/progress.md` (Source for dual-model and Phi-3-mini info)
*   `autonomous_agent/config.py` (Source for exact model names)

## IMPLEMENTATION DETAILS
**1. `autonomous_agent/main.py` Updates:**
    *   Modify the `install_model` function:
        *   Change `ollama pull wizard-vicuna` to pull `deepseek-coder:6.7b-instruct`.
        *   Add a command to also pull `phi3:mini`.
        *   Update print statements to reflect these changes.

**2. `README.md` Updates:**
    *   Modify "Overview" to state `deepseek-coder:6.7b-instruct` is used and mention the dual-model plan with `phi3:mini`.
    *   Update "Key Features" > "Local LLM Execution" to name `deepseek-coder:6.7b-instruct`.
    *   Update "Technology Stack" > "LLM Engine" to `Ollama (with deepseek-coder:6.7b-instruct as primary coding model; phi3:mini as reasoning model for dual-model architecture)`.
    *   Update "Installation" > "Pull/Set up the LLM Model" to reflect the changes in `install_model` command, guiding to install `deepseek-coder:6.7b-instruct` and `phi3:mini`.
    *   Verify "Development Roadmap" and "Current Development Focus" for consistency with dual-model plans.
    *   Remove/replace outdated mentions of `Wizard-Vicuna-13B-Uncensored`.

**3. `dev_log/summary.md` Updates:**
    *   Update "Overview" section to replace `Wizard-Vicuna-13B-Uncensored` with `deepseek-coder:6.7b-instruct` and briefly mention the dual-model strategy with `phi3:mini`.
    *   Update "Model Interface" section to reflect testing/use with `deepseek-coder:6.7b-instruct`.
    *   Remove or update the `Wizard-Vicuna-13B-Uncensored.Q4_K_M.gguf` entry from the project structure diagram or clarify its deprecated status.
    *   Update "Recent Improvements" > "Verified Model Integration" to reflect successful testing with `deepseek-coder:6.7b-instruct`.

## TODO LIST NAME: Update Project Models Documentation and Scripts
*   [x] Read `autonomous_agent/main.py` and identify the `install_model` function.
*   [x] Modify `install_model` in `autonomous_agent/main.py` to pull `deepseek-coder:6.7b-instruct` and `phi3:mini`, and update print statements.
*   [x] Read current `README.md`.
*   [x] Update "Overview" in `README.md`.
*   [x] Update "Key Features" in `README.md`.
*   [x] Update "Technology Stack" in `README.md`.
*   [x] Update "Installation" instructions in `README.md` based on `main.py` changes.
*   [x] Verify "Development Roadmap" & "Current Development Focus" in `README.md`.
*   [x] Remove/correct outdated `Wizard-Vicuna` references in `README.md`.
*   [x] Save updated `README.md`.
*   [x] Read `dev_log/summary.md`.
*   [x] Update "Overview" in `dev_log/summary.md`.
*   [x] Update "Model Interface" section in `dev_log/summary.md`.
*   [x] Update project structure diagram in `dev_log/summary.md` regarding model file.
*   [x] Update "Recent Improvements" in `dev_log/summary.md`.
*   [x] Save updated `dev_log/summary.md`.
*   [x] Create the task file `_context/tasks/update_readme_models_v2.md` with this plan.

## MEETING NOTES
- Task initiated to update project documentation (`README.md`, `dev_log/summary.md`) and scripts (`autonomous_agent/main.py`) to reflect the current model setup (DeepSeek-Coder and Phi-3-mini) and deprecate Wizard-Vicuna references.
- All modifications completed as per the plan. Files updated and ready for review/commit. 