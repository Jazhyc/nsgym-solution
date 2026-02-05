# Evaluation image -- edit this file to add custom non-Python dependencies
FROM ns-gym-base

# ---- Add custom dependencies below this line ----
# System packages:  RUN apt-get update && apt-get install -y <package>
# Python packages:  RUN uv pip install <package>

COPY src/ ./src/
RUN uv pip install -e . --no-deps

COPY evaluator.py .
COPY submission.py .

CMD ["uv", "run", "python", "evaluator.py", "--mode", "local"]
