import gc
import os
import time

from llama_cpp import Llama  # pyright: ignore[reportMissingTypeStubs]

from exo.download.download_utils import build_model_path
from exo.shared.types.api import (
    CompletionTokensDetails,
    GenerationStats,
    PromptTokensDetails,
    Usage,
)
from exo.shared.types.chunks import ErrorChunk, TokenChunk
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from exo.shared.types.memory import Memory
from exo.shared.types.tasks import (
    ConnectToGroup,
    LoadModel,
    Shutdown,
    StartWarmup,
    Task,
    TaskId,
    TaskStatus,
    TextGeneration,
)
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runners import (
    RunnerConnected,
    RunnerIdle,
    RunnerLoaded,
    RunnerLoading,
    RunnerReady,
    RunnerRunning,
    RunnerShutdown,
    RunnerShuttingDown,
    RunnerStatus,
    RunnerWarmingUp,
)
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.runner.bootstrap import logger

from .utils import build_chat_messages, find_gguf_file


def main(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event],
    task_receiver: MpReceiver[Task],
    cancel_receiver: MpReceiver[TaskId],
) -> None:
    runner_id = bound_instance.bound_runner_id
    shard_metadata = bound_instance.bound_shard
    model_id = shard_metadata.model_card.model_id

    logger.info("hello from the llamacpp runner")

    setup_start_time = time.time()
    cancelled_tasks: set[TaskId] = set()

    llm: Llama | None = None

    current_status: RunnerStatus = RunnerIdle()
    logger.info("llamacpp runner created")
    event_sender.send(
        RunnerStatusUpdated(runner_id=runner_id, runner_status=current_status)
    )

    seen: set[TaskId] = set()
    with task_receiver as tasks:
        for task in tasks:
            if task.task_id in seen:
                logger.warning("repeat task - potential error")
            seen.add(task.task_id)
            cancelled_tasks.discard(TaskId("CANCEL_CURRENT_TASK"))
            event_sender.send(
                TaskStatusUpdated(task_id=task.task_id, task_status=TaskStatus.Running)
            )

            match task:
                case ConnectToGroup():
                    # No distributed support — just transition through the states
                    logger.info("llamacpp runner: connect (no-op for single-node)")
                    current_status = RunnerConnected()
                    event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=runner_id, runner_status=current_status
                        )
                    )
                    event_sender.send(TaskAcknowledged(task_id=task.task_id))

                case LoadModel() if isinstance(
                    current_status, (RunnerConnected, RunnerIdle)
                ):
                    current_status = RunnerLoading()
                    logger.info("llamacpp runner loading model")
                    event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=runner_id, runner_status=current_status
                        )
                    )
                    event_sender.send(TaskAcknowledged(task_id=task.task_id))

                    model_path = build_model_path(model_id)
                    # model_path may be a directory or a direct .gguf file
                    if model_path.is_file() and model_path.suffix == ".gguf":
                        gguf_path = model_path
                    else:
                        gguf_path = find_gguf_file(model_path)

                    n_ctx = int(os.environ.get("EXO_LLAMACPP_N_CTX", "4096"))
                    cpu_count = os.cpu_count() or 4
                    n_threads = int(os.environ.get("EXO_LLAMACPP_N_THREADS", str(max(1, cpu_count // 2))))
                    n_threads_batch = int(os.environ.get("EXO_LLAMACPP_N_THREADS_BATCH", str(cpu_count)))
                    n_batch = int(os.environ.get("EXO_LLAMACPP_N_BATCH", "512"))
                    flash_attn = os.environ.get("EXO_LLAMACPP_FLASH_ATTN", "1") == "1"
                    type_k = int(os.environ.get("EXO_LLAMACPP_TYPE_K", "8"))   # GGML_TYPE_Q8_0
                    type_v = int(os.environ.get("EXO_LLAMACPP_TYPE_V", "8"))   # GGML_TYPE_Q8_0

                    logger.info(
                        f"Loading GGUF model from {gguf_path} "
                        f"(n_ctx={n_ctx}, n_threads={n_threads}, n_threads_batch={n_threads_batch}, "
                        f"n_batch={n_batch}, flash_attn={flash_attn}, type_k={type_k}, type_v={type_v})"
                    )
                    load_start = time.monotonic()
                    llm = Llama(
                        model_path=str(gguf_path),
                        n_ctx=n_ctx,
                        n_threads=n_threads,
                        n_threads_batch=n_threads_batch,
                        n_batch=n_batch,
                        flash_attn=flash_attn,
                        type_k=type_k,
                        type_v=type_v,
                        use_mmap=True,
                        verbose=False,
                    )  # pyright: ignore[reportUnknownMemberType]
                    logger.info(f"llamacpp model loaded in {time.monotonic() - load_start:.1f}s")
                    current_status = RunnerLoaded()

                case StartWarmup() if isinstance(current_status, RunnerLoaded):
                    current_status = RunnerWarmingUp()
                    logger.info("llamacpp runner warming up")
                    event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=runner_id, runner_status=current_status
                        )
                    )
                    event_sender.send(TaskAcknowledged(task_id=task.task_id))

                    assert llm is not None
                    # Quick warmup: generate a few tokens
                    llm.create_chat_completion(  # pyright: ignore[reportUnknownMemberType]
                        messages=[{"role": "user", "content": "Hi"}],
                        max_tokens=4,
                    )
                    logger.info(
                        f"llamacpp runner warmed up in "
                        f"{time.time() - setup_start_time:.1f}s"
                    )
                    current_status = RunnerReady()
                    logger.info("llamacpp runner ready")

                case TextGeneration(
                    task_params=task_params, command_id=command_id
                ) if isinstance(current_status, RunnerReady):
                    logger.info(f"llamacpp received chat request: {task}")
                    current_status = RunnerRunning()
                    event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=runner_id, runner_status=current_status
                        )
                    )
                    event_sender.send(TaskAcknowledged(task_id=task.task_id))
                    assert llm is not None

                    try:
                        messages = build_chat_messages(task_params)
                        max_tokens = task_params.max_output_tokens or 2048
                        temperature = task_params.temperature or 0.7
                        top_p = task_params.top_p or 1.0

                        gen_start = time.monotonic()
                        first_token_time: float | None = None
                        completion_tokens = 0
                        prompt_tokens = 0

                        stream = llm.create_chat_completion(  # pyright: ignore[reportUnknownMemberType]
                            messages=messages,  # pyright: ignore[reportArgumentType]
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            stream=True,
                        )

                        for chunk in stream:  # pyright: ignore[reportUnknownVariableType]
                            # Check for cancellation
                            cancelled_tasks.update(cancel_receiver.collect())
                            if (task.task_id in cancelled_tasks) or (
                                TaskId("CANCEL_CURRENT_TASK") in cancelled_tasks
                            ):
                                break

                            choices = chunk.get("choices", [])  # pyright: ignore[reportUnknownMemberType]
                            if not choices:  # pyright: ignore[reportUnknownArgumentType]
                                # Try to capture prompt_tokens from usage in non-choice chunks
                                chunk_usage = chunk.get("usage")  # pyright: ignore[reportUnknownMemberType]
                                if chunk_usage:  # pyright: ignore[reportUnknownArgumentType]
                                    prompt_tokens = chunk_usage.get("prompt_tokens", prompt_tokens)  # pyright: ignore[reportUnknownMemberType]
                                continue

                            choice = choices[0]  # pyright: ignore[reportUnknownVariableType]
                            delta = choice.get("delta", {})  # pyright: ignore[reportUnknownMemberType]
                            content = delta.get("content", "")  # pyright: ignore[reportUnknownMemberType]
                            finish_reason_raw = choice.get("finish_reason")  # pyright: ignore[reportUnknownMemberType]

                            if content:  # pyright: ignore[reportUnknownArgumentType]
                                if first_token_time is None:
                                    first_token_time = time.monotonic()
                                completion_tokens += 1

                            # Map finish_reason
                            finish_reason = None
                            if finish_reason_raw == "stop":
                                finish_reason = "stop"
                            elif finish_reason_raw == "length":
                                finish_reason = "length"

                            # Build usage on final chunk
                            usage = None
                            stats = None
                            if finish_reason is not None:
                                now = time.monotonic()
                                # Prompt TPS: tokens processed before first generated token
                                prompt_elapsed = (first_token_time or now) - gen_start
                                prompt_tps = prompt_tokens / prompt_elapsed if prompt_elapsed > 0 and prompt_tokens > 0 else 0.0
                                # Generation TPS: tokens generated after first token
                                gen_elapsed = now - (first_token_time or gen_start)
                                gen_tps = (
                                    completion_tokens / gen_elapsed if gen_elapsed > 0 else 0.0
                                )
                                logger.info(
                                    f"llamacpp generation done: {prompt_tokens} prompt tokens @ {prompt_tps:.1f} t/s, "
                                    f"{completion_tokens} tokens @ {gen_tps:.1f} t/s"
                                )
                                usage = Usage(
                                    prompt_tokens=prompt_tokens,
                                    completion_tokens=completion_tokens,
                                    total_tokens=prompt_tokens + completion_tokens,
                                    prompt_tokens_details=PromptTokensDetails(),
                                    completion_tokens_details=CompletionTokensDetails(),
                                )
                                stats = GenerationStats(
                                    prompt_tps=prompt_tps,
                                    generation_tps=gen_tps,
                                    prompt_tokens=prompt_tokens,
                                    generation_tokens=completion_tokens,
                                    peak_memory_usage=Memory(in_bytes=0),
                                )

                            if content or finish_reason is not None:  # pyright: ignore[reportUnknownArgumentType]
                                event_sender.send(
                                    ChunkGenerated(
                                        command_id=command_id,
                                        chunk=TokenChunk(
                                            model=model_id,
                                            text=str(content or ""),
                                            token_id=0,
                                            usage=usage,
                                            finish_reason=finish_reason,
                                            stats=stats,
                                        ),
                                    )
                                )

                    except Exception as e:
                        event_sender.send(
                            ChunkGenerated(
                                command_id=command_id,
                                chunk=ErrorChunk(
                                    model=model_id,
                                    error_message=str(e),
                                ),
                            )
                        )
                        raise

                    current_status = RunnerReady()
                    logger.info("llamacpp runner ready")

                case Shutdown():
                    current_status = RunnerShuttingDown()
                    logger.info("llamacpp runner shutting down")
                    del llm
                    llm = None
                    gc.collect()

                    event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=runner_id, runner_status=current_status
                        )
                    )
                    event_sender.send(TaskAcknowledged(task_id=task.task_id))
                    current_status = RunnerShutdown()

                case _:
                    raise ValueError(
                        f"Received {task.__class__.__name__} outside of state machine "
                        f"in {current_status=}"
                    )

            was_cancelled = (task.task_id in cancelled_tasks) or (
                TaskId("CANCEL_CURRENT_TASK") in cancelled_tasks
            )
            if not was_cancelled:
                event_sender.send(
                    TaskStatusUpdated(
                        task_id=task.task_id, task_status=TaskStatus.Complete
                    )
                )
            event_sender.send(
                RunnerStatusUpdated(runner_id=runner_id, runner_status=current_status)
            )

            if isinstance(current_status, RunnerShutdown):
                break
