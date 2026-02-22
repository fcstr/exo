from pathlib import Path

from exo.shared.types.text_generation import TextGenerationTaskParams


def find_gguf_file(model_dir: Path) -> Path:
    """Find the first .gguf file in a model directory."""
    gguf_files = sorted(model_dir.glob("*.gguf"))
    if not gguf_files:
        raise FileNotFoundError(f"No .gguf file found in {model_dir}")
    return gguf_files[0]


def build_chat_messages(task_params: TextGenerationTaskParams) -> list[dict[str, str]]:
    """Convert TextGenerationTaskParams to llama-cpp-python chat message format."""
    messages: list[dict[str, str]] = []
    if task_params.instructions:
        messages.append({"role": "system", "content": task_params.instructions})
    for msg in task_params.input:
        messages.append({"role": msg.role, "content": msg.content})
    return messages
