You are working on a Python repository for a knowledge-graph-based fact-checking system.

Your task is to update and significantly improve the project documentation so that the repository can be handed over to a supervisor/tutor.

## Main goal

Rewrite the README so it accurately reflects the CURRENT CODEBASE, not the old documentation.

The theoretical part of the project is already explained elsewhere in a separate report.  
So in this repository, the documentation must focus on:

1. how the codebase is organized,
2. how the code works at a high level,
3. how to set up the environment,
4. how to run and use the repository,
5. what each module/file does,
6. what the main functions do,
7. what the current limitations are.

## Important constraints

- Everything in the documentation must be written in English.
- Any code comments, docstrings, or inline explanations that are not in English must be rewritten in English.
- The README must explain clearly that **almost everything goes through `cli.py`**.
- It must also explain that **`build_evaluate.py` at the repository root is the exception**:
  - it is currently experimental / still under testing,
  - it already produces the desired benchmark output,
  - but it was not fully finalized due to time constraints,
  - therefore it should be documented honestly as work-in-progress.

## What I want you to do

### 1. Inspect the repository structure
Read the repository and infer the real workflow from the code itself.  
Do not trust the current README if it differs from the implementation.

### 2. Rewrite the README
Produce a new README that is:
- clear,
- professional,
- detailed,
- accurate,
- easy for another person to use without prior context.

The README should include at least the following sections:

- Project title
- Short project description
- Repository purpose
- High-level architecture / workflow
- Repository structure
- Installation / environment setup
- Dependencies and prerequisites
- Configuration files and how they are used
- How to initialize the project
- How to use `cli.py`
- Detailed explanation of the main CLI commands
- Inputs / outputs
- Generated files and where they are stored
- Explanation of each main Python module
- Explanation of the experimental `build_evaluate.py`
- Typical usage examples
- Current limitations / known issues

### 3. Document each Python file
For each `.py` file in the repo:
- add or improve the module-level docstring,
- explain the role of the file,
- document the important functions,
- document classes if any,
- add clean English docstrings where missing,
- improve unclear names or comments only if necessary and safe.

Do not make unnecessary refactors unless needed for clarity.

### 4. Explain the execution flow
The documentation must make the execution flow very explicit, especially:
- the user usually starts from `cli.py`,
- `cli.py` dispatches commands to the relevant modules,
- explain which modules are involved in build / retrieval / visualization / config / evaluation,
- explain the role of the root-level `build_evaluate.py` separately.

### 5. Environment setup
Add a complete setup section:
- Python version
- package manager / install commands
- virtual environment instructions if relevant
- dependency installation
- extra model downloads if needed (e.g. spaCy models, Ollama model if used)
- configuration steps
- database setup if relevant
- initialization commands

### 6. Keep the documentation practical
The README should prioritize:
- how to run the code,
- how to understand the repo quickly,
- how to use the available features,
- how outputs are produced.

Avoid re-explaining the full theory behind BERTopic / NER / retrieval unless necessary for understanding usage.

## Expected output

I want you to produce:

1. a fully rewritten `README.md`,
2. improved English docstrings/comments across the codebase,
3. concise documentation for each Python module and its main functions,
4. a short summary of what you changed,
5. a note listing any parts of the repo whose behavior was unclear or inconsistent with the previous README.

## Style requirements for the README

- Use professional technical English.
- Be explicit and concrete.
- Prefer practical explanations over academic theory.
- Use consistent headings and formatting.
- Include command examples in code blocks.
- Make it easy for a tutor/supervisor to understand the repo quickly.

## Additional guidance

- Read the actual code before writing.
- If the old README contains outdated information, replace it with what the code really does.
- If a feature is incomplete, say so clearly and honestly.
- Do not invent functionality that is not implemented.
- If some parts are ambiguous, document them conservatively.

Start by analyzing the repository structure and the real entry points, then rewrite the documentation accordingly.