import os
import platform

import loguru

from exo.shared.types.events import Event, RunnerStatusUpdated
from exo.shared.types.tasks import Task, TaskId
from exo.shared.types.worker.instances import BoundInstance, MlxJacclInstance
from exo.shared.types.worker.runners import RunnerFailed
from exo.utils.channels import ClosedResourceError, MpReceiver, MpSender

logger: "loguru.Logger" = loguru.logger


def _should_use_llamacpp() -> bool:
    """Check if we should use the llama.cpp backend.

    Uses EXO_INFERENCE_ENGINE=llamacpp env var, or auto-detects Linux + aarch64.
    """
    engine = os.environ.get("EXO_INFERENCE_ENGINE", "").lower()
    if engine == "llamacpp":
        return True
    if engine and engine != "llamacpp":
        return False
    # Auto-detect: Linux + aarch64 (e.g. Android/proot)
    return platform.system() == "Linux" and platform.machine() == "aarch64"


def entrypoint(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event],
    task_receiver: MpReceiver[Task],
    cancel_receiver: MpReceiver[TaskId],
    _logger: "loguru.Logger",
) -> None:
    use_llamacpp = _should_use_llamacpp()

    if not use_llamacpp:
        fast_synch_override = os.environ.get("EXO_FAST_SYNCH")
        if fast_synch_override == "on" or (
            fast_synch_override != "off"
            and (
                isinstance(bound_instance.instance, MlxJacclInstance)
                and len(bound_instance.instance.jaccl_devices) >= 2
            )
        ):
            os.environ["MLX_METAL_FAST_SYNCH"] = "1"
        else:
            os.environ["MLX_METAL_FAST_SYNCH"] = "0"

    global logger
    logger = _logger

    if not use_llamacpp:
        logger.info(f"Fast synch flag: {os.environ['MLX_METAL_FAST_SYNCH']}")
    else:
        logger.info("Using llama.cpp inference backend")

    # Import main after setting global logger - this lets us just import logger from this module
    try:
        if use_llamacpp:
            from exo.worker.runner.llamacpp_inference.runner import main
        elif bound_instance.is_image_model:
            from exo.worker.runner.image_models.runner import main
        else:
            from exo.worker.runner.llm_inference.runner import main

        main(bound_instance, event_sender, task_receiver, cancel_receiver)

    except ClosedResourceError:
        logger.warning("Runner communication closed unexpectedly")
    except Exception as e:
        logger.opt(exception=e).warning(
            f"Runner {bound_instance.bound_runner_id} crashed with critical exception {e}"
        )
        event_sender.send(
            RunnerStatusUpdated(
                runner_id=bound_instance.bound_runner_id,
                runner_status=RunnerFailed(error_message=str(e)),
            )
        )
    finally:
        try:
            event_sender.close()
            task_receiver.close()
        finally:
            event_sender.join()
            task_receiver.join()
            logger.info("bye from the runner")
