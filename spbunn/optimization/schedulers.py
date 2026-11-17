from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class AnnotationSettings:
    text: str
    xy: tuple[float, float]
    xytext: tuple[float, float]


@dataclass(frozen=True)
class ScheduleSettings:
    name: str
    title: str
    schedule_type: str
    annotations: tuple[AnnotationSettings, ...]
    decrement: float = 0.003
    alpha: float = 0.5
    step_size: int = 5
    last_epoch: int = 25
    repetitions: int = 4


@dataclass(frozen=True)
class SchedulerInput:
    epochs: IntArray
    starting_lr: float
    schedules: tuple[ScheduleSettings, ...]


@dataclass(frozen=True)
class ScheduleCurve:
    settings: ScheduleSettings
    epochs: IntArray
    learning_rates: FloatArray


def _point(values: Iterable[float]) -> tuple[float, float]:
    point = tuple(float(value) for value in values)
    if len(point) != 2:
        raise ValueError("Annotation coordinates must contain two values")
    return point[0], point[1]


def produce_scheduler_input(cfg: DictConfig) -> SchedulerInput:
    """Create the epoch grid and typed scheduler settings."""
    n_epochs = int(cfg.n_epochs)
    if n_epochs <= 0:
        raise ValueError("The epoch count must be positive")
    schedules = []
    for schedule in cfg.schedulers:
        annotations = tuple(
            AnnotationSettings(
                text=str(annotation.text),
                xy=_point(annotation.xy),
                xytext=_point(annotation.xytext),
            )
            for annotation in (schedule.annotations or ())
        )
        schedules.append(
            ScheduleSettings(
                name=str(schedule.name),
                title=str(schedule.title),
                schedule_type=str(schedule.schedule_type),
                annotations=annotations,
                decrement=float(schedule.get("decrement", 0.003)),
                alpha=float(schedule.get("alpha", 0.5)),
                step_size=int(schedule.get("step_size", 5)),
                last_epoch=int(schedule.get("last_epoch", 25)),
                repetitions=int(schedule.get("repetitions", 4)),
            )
        )
    return SchedulerInput(
        epochs=np.arange(n_epochs, dtype=np.int64),
        starting_lr=float(cfg.starting_lr),
        schedules=tuple(schedules),
    )


def _linear_schedule(
    settings: ScheduleSettings, experiment: SchedulerInput
) -> FloatArray:
    epochs = experiment.epochs
    transition = len(epochs) // 2 - 1
    if transition < 0:
        raise ValueError("The linear schedule needs at least two epochs")
    learning_rates = experiment.starting_lr - settings.decrement * epochs
    learning_rates[transition + 1 :] = learning_rates[transition]
    return np.asarray(learning_rates, dtype=np.float64)


def _step_schedule(
    settings: ScheduleSettings, experiment: SchedulerInput
) -> FloatArray:
    if settings.alpha <= 0:
        raise ValueError("Step-scheduler alpha must be positive")
    if settings.step_size <= 0 or settings.last_epoch <= 0:
        raise ValueError("Step-scheduler epoch settings must be positive")
    epochs = experiment.epochs
    learning_rates = np.empty(len(epochs), dtype=np.float64)
    epoch = 0
    current_lr = experiment.starting_lr / settings.alpha
    while epoch < min(settings.last_epoch, len(epochs)):
        end = min(epoch + settings.step_size, len(epochs))
        learning_rates[epoch:end] = current_lr
        epoch = end
        current_lr *= settings.alpha
    if epoch < len(epochs):
        learning_rates[epoch:] = current_lr / settings.alpha
    return learning_rates


def _plateau_schedule(
    settings: ScheduleSettings, experiment: SchedulerInput
) -> FloatArray:
    n_epochs = len(experiment.epochs)
    if n_epochs < 4:
        raise ValueError("The reduce-on-plateau schedule needs at least four epochs")
    first_end = n_epochs // 4
    second_end = 3 * n_epochs // 4
    return np.concatenate(
        [
            np.full(first_end, experiment.starting_lr),
            np.full(second_end - first_end, settings.alpha * experiment.starting_lr),
            np.full(
                n_epochs - second_end,
                settings.alpha**2 * experiment.starting_lr,
            ),
        ]
    ).astype(np.float64)


def _cosine_schedule(
    settings: ScheduleSettings, experiment: SchedulerInput
) -> FloatArray:
    if settings.repetitions <= 0:
        raise ValueError("Cosine-scheduler repetitions must be positive")
    n_epochs = len(experiment.epochs)
    if n_epochs % settings.repetitions != 0:
        raise ValueError("The epoch count must be divisible by cosine repetitions")
    quarter_length = n_epochs // settings.repetitions
    if quarter_length == 0:
        raise ValueError("The cosine schedule needs a non-empty quarter")
    starting_lr = experiment.starting_lr
    quarter = starting_lr + 0.5 * (starting_lr - 0.5 * starting_lr) * (
        1
        + np.cos(
            experiment.epochs[:quarter_length] / quarter_length * np.pi,
        )
    )
    return np.tile(quarter, settings.repetitions).astype(np.float64)


def evaluate_schedules(experiment: SchedulerInput) -> tuple[ScheduleCurve, ...]:
    """Evaluate all configured schedules without rendering or file output."""
    evaluators = {
        "linear": _linear_schedule,
        "step": _step_schedule,
        "reduce_on_plateau": _plateau_schedule,
        "cosine_annealing": _cosine_schedule,
    }
    curves = []
    for settings in experiment.schedules:
        try:
            evaluator = evaluators[settings.schedule_type]
        except KeyError as error:
            raise ValueError(
                f"Unsupported schedule: {settings.schedule_type}"
            ) from error
        curves.append(
            ScheduleCurve(
                settings=settings,
                epochs=experiment.epochs,
                learning_rates=evaluator(settings, experiment),
            )
        )
    return tuple(curves)
