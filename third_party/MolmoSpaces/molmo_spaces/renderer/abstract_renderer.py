import abc
import threading
from collections import Counter, defaultdict
from collections.abc import Callable
from queue import Queue
from typing import TYPE_CHECKING, Any, NoReturn

from mujoco import MjModel

from molmo_spaces.env.mj_extensions import MjModelBindings

if TYPE_CHECKING:
    from molmo_spaces.env.vector_env import MuJoCoVectorEnv

RENDERING_COMPLETE = "RENDERING_COMPLETE"


class MjAbstractRenderer(abc.ABC):
    render_outputs: list[Any]

    def __init__(
        self,
        model_bindings: MjModelBindings = None,
        device_id: int | None = None,
        model: MjModel = None,
        **kwargs,
    ) -> None:
        self._model_bindings = model_bindings
        self._model = model
        self.device_id = device_id

    @property
    def model(self):
        return self._model  # or self._model_bindings.model

    @property
    def model_bindings(self):
        return self._model_bindings

    @abc.abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def render(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def reset_single(self, idx: int) -> None:
        return None


class MultithreadRenderer(abc.ABC):
    def __init__(
        self,
        env: "MuJoCoVectorEnv",
        renderer_cls: type[MjAbstractRenderer],
        max_render_contexts: int | None = 1_000_000,
        process_request_kwargs: dict | None = None,
        **additional_rendering_thread_runner_kwargs: Any,
    ) -> None:
        self._closed = False

        self.env = env
        self.max_render_contexts = max_render_contexts or 1

        self.model_id_to_render_threads: dict[int, list[threading.Thread]] | None = None
        self.model_id_to_render_input_queue: dict[int, Queue] | None = None
        self.render_output_queue: Queue | None = None

        model_id_to_count = Counter(id(model) for model in env.mj_models)

        if self.max_render_contexts > 0:
            assert self.max_render_contexts >= len(model_id_to_count), (
                f"max_render_contexts ({self.max_render_contexts}) must be greater than or equal to "
                f"the number of unique models ({len(model_id_to_count)})"
            )

        self.model_id_to_render_threads = defaultdict(list)
        self.model_id_to_render_input_queue = defaultdict(Queue)
        self.render_output_queue = Queue()

        num_render_contexts = min(self.max_render_contexts, sum(model_id_to_count.values()))
        num_render_threads_started = 0

        while num_render_threads_started < num_render_contexts:
            for model_id, count in list(model_id_to_count.items()):
                if count > 0:
                    model_id_to_count[model_id] -= 1

                    self.model_id_to_render_threads[model_id].append(
                        threading.Thread(
                            target=self.rendering_thread_runner,
                            kwargs=dict(
                                renderer_cls=renderer_cls,
                                model_bindings=env.model_id_to_model_container[model_id],
                                device=env.device,
                                input_queue=self.model_id_to_render_input_queue[model_id],
                                output_queue=self.render_output_queue,
                                process_request_callback=self.process_request,
                                process_request_kwargs=process_request_kwargs,
                                **additional_rendering_thread_runner_kwargs,
                            ),
                        )
                    )
                    self.model_id_to_render_threads[model_id][-1].start()
                    num_render_threads_started += 1

                    if num_render_threads_started == num_render_contexts:
                        return

    @staticmethod
    @abc.abstractmethod
    def process_request(
        renderer: MjAbstractRenderer, request: Any, output_queue: Queue, **kwargs
    ) -> NoReturn:
        raise NotImplementedError

    @staticmethod
    def rendering_thread_runner(
        renderer_cls: type[MjAbstractRenderer],
        process_request_callback: Callable[[MjAbstractRenderer, Any, Queue, dict | None], None],
        model_bindings: MjModelBindings,
        device: int | None,
        input_queue: Queue,
        output_queue: Queue,
        timeout: int | None = None,
        process_request_kwargs: dict | None = None,
        **kwargs: Any,
    ) -> None:
        renderer = renderer_cls(model_bindings=model_bindings, device_id=device, **kwargs)

        print(
            f"Rendering thread started with renderer {renderer_cls.__name__} for model {id(model_bindings.model)}"
        )
        process_request_kwargs = process_request_kwargs or {}
        try:
            while True:
                request = input_queue.get(block=True, timeout=timeout)

                if request == RENDERING_COMPLETE:
                    print(
                        f"Rendering thread for model {id(model_bindings.model)} received RENDERING_COMPLETE"
                    )
                    break

                process_request_callback(renderer, request, output_queue, **process_request_kwargs)
        finally:
            renderer.close()

    def __del__(self) -> None:
        if self._closed:
            return

        self._closed = True

        try:
            for _idx, model in enumerate(self.env.mj_models):
                self.model_id_to_render_input_queue[id(model)].put(RENDERING_COMPLETE)

            for threads in self.model_id_to_render_threads.values():
                for thread in threads:
                    thread.join(0.1)

            self.model_id_to_render_threads.clear()
            self.model_id_to_render_input_queue.clear()
        except (KeyboardInterrupt, SystemExit):
            raise
        except ValueError:
            print("While closing MultithreadRenderer")

    @abc.abstractmethod
    def render(self, *args, **kwargs) -> Any:
        raise NotImplementedError
