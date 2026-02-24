from pathlib import Path

from exo.shared.types.text_generation import TextGenerationTaskParams


_GGUF_PREFERENCE = ["q4_k_m", "q4_k_s", "q4_0", "q5_k_m", "q3_k_m", "q8_0", "q6_k", "q5_0", "q2_k"]


def find_gguf_file(model_dir: Path) -> Path:
    """Find the best .gguf file in a model directory.

    Prefers quantized models (q4_k_m > q4_k_s > ...) over fp16/fp32.
    Override with EXO_LLAMACPP_GGUF env var (substring match on filename).
    """
    import os

    gguf_files = sorted(model_dir.glob("*.gguf"))
    if not gguf_files:
        raise FileNotFoundError(f"No .gguf file found in {model_dir}")

    override = os.environ.get("EXO_LLAMACPP_GGUF", "").lower()
    if override:
        for f in gguf_files:
            if override in f.name.lower():
                return f

    for pref in _GGUF_PREFERENCE:
        for f in gguf_files:
            if pref in f.name.lower():
                return f

    return gguf_files[0]


def build_chat_messages(task_params: TextGenerationTaskParams) -> list[dict[str, str]]:
    """Convert TextGenerationTaskParams to llama-cpp-python chat message format."""
    messages: list[dict[str, str]] = []
    if task_params.instructions:
        messages.append({"role": "system", "content": task_params.instructions})
    for msg in task_params.input:
        messages.append({"role": msg.role, "content": msg.content})
    return messages
