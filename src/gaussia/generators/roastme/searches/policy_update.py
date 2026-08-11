"""The training-backed policy update: one clipped policy-gradient step on the category generator.

Needs ``gaussia[roastme-rl]`` and a GPU. It is the only module of the subsystem that imports the
training stack, which is what lets ``policy_gradient.py`` — the loop that calls it — run in the
default suite on CPU with neither (FR-033, FR-037, SC-014).

The extra is not imported defensively. Without it, importing this module raises a plain
``ImportError`` naming the package that is missing, at the line that needs it. A ``try/except
ImportError`` here would turn a missing dependency into a fault surfacing later, somewhere else,
under a message gaussia invented; a dynamic import would move the same failure to the first update
step, after a run has already spent its target calls.

The update is the clipped surrogate of PPO over a bandit reward: one step per iteration, the batch
mean as the baseline, and the importance ratio taken against the log-probability the policy
reported when it sampled. That ratio is why the search passes the log-probability through instead
of recomputing it — recomputed here it would be the *updated* policy's, and the correction it is
meant to apply would vanish. There is no value head and no reference model: a category is scored
once, in full, so there is nothing to bootstrap a value function from, and the clip is the whole
trust region.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import torch
from accelerate import Accelerator

from .policy_gradient import PolicyUpdateStep

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from peft import PeftModel
    from transformers import PreTrainedTokenizerBase

    from gaussia.schemas.roastme import Category

    from .policy_gradient import RewardedCandidate

_LOGITS_OF_THE_NEXT_TOKEN = slice(None, -1)
_TOKENS_AFTER_THE_FIRST = slice(1, None)


class ClipHyperparameters(Protocol):
    """The two hyperparameters the step reads, named structurally rather than by trl's class.

    trl's ``PPOConfig`` satisfies this and is still what to pass — but it has moved between trl's
    top level and ``trl.experimental.ppo`` across the versions ``gaussia[roastme-rl]`` permits, so
    no single import path is correct for the declared range. This module never calls trl: the
    clipped surrogate is computed here. Typing against the two fields actually read keeps the
    annotation true whatever trl does with its own layout next.

    A ``Protocol`` rather than an ABC because this describes the shape of somebody else's object,
    not an interface a user implements against — the ten of those are in ``core/``.
    """

    cliprange: float
    learning_rate: float


class ClippedPolicyUpdate(PolicyUpdateStep):
    """Applies one PPO-clipped policy-gradient step to the adapters of the category policy.

    Args:
        model: The policy's language model, already wrapped in the adapters that carry the
            training. It must be the same object the ``CategoryPolicy`` samples from — the step
            optimises what that policy will sample next, and two copies would leave the search
            reporting one model's categories while training another's. Wrapped rather than
            fine-tuned whole because only the adapters need gradients, which is what makes a step
            fit on one GPU.
        tokenizer: The tokenizer of ``model``.
        render: How the policy turned a category into the text it sampled. Injected because the
            log-probability recomputed here has to be of the same string the reported one was of,
            and only the policy knows how it wrote it.
        config: The clip range and the step size. Pass trl's ``PPOConfig`` so both are the
            reference implementation's rather than numbers invented here; anything carrying the two
            fields will do.
    """

    def __init__(
        self,
        model: PeftModel,
        tokenizer: PreTrainedTokenizerBase,
        render: Callable[[Category], str],
        config: ClipHyperparameters,
    ) -> None:
        self._tokenizer = tokenizer
        self._render = render
        self._cliprange = config.cliprange
        self._accelerator = Accelerator()
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable:
            message = (
                "the policy's model carries no trainable parameter, so a gradient step would be a "
                "no-op; wrap it in its adapters before handing it over"
            )
            raise ValueError(message)
        self._model = self._accelerator.prepare_model(model)
        self._optimiser = self._accelerator.prepare_optimizer(torch.optim.AdamW(trainable, lr=config.learning_rate))

    def apply(self, samples: list[RewardedCandidate]) -> None:
        if not samples:
            return
        rewards = self._tensor([reward for _, _, reward in samples])
        # The batch mean as the baseline: it centres the rewards without a value function, so a
        # candidate is pushed up only for beating the batch it was drawn with rather than for
        # scoring above zero, which every candidate that survived the gates does.
        advantages = rewards - rewards.mean()
        sampled = self._tensor([log_probability for _, log_probability, _ in samples])
        ratio = torch.exp(self._log_probabilities([category for category, _, _ in samples]) - sampled)
        clipped = torch.clamp(ratio, 1.0 - self._cliprange, 1.0 + self._cliprange)
        loss = -torch.min(ratio * advantages, clipped * advantages).mean()
        self._optimiser.zero_grad()
        self._accelerator.backward(loss)
        self._optimiser.step()

    def _tensor(self, values: Sequence[float]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.float32, device=self._accelerator.device)

    def _log_probabilities(self, categories: Sequence[Category]) -> torch.Tensor:
        """The log-probability the current policy assigns each category's text, summed over tokens.

        Padding is masked out rather than trusted to contribute nothing: a pad token has a
        log-probability like any other, and leaving it in would score a short category against how
        confidently the model predicts padding.
        """
        batch = self._tokenizer(
            [self._render(category) for category in categories],
            return_tensors="pt",
            padding=True,
        ).to(self._accelerator.device)
        logits = self._model(**batch).logits[:, _LOGITS_OF_THE_NEXT_TOKEN]
        tokens = batch["input_ids"][:, _TOKENS_AFTER_THE_FIRST]
        per_token = torch.log_softmax(logits, dim=-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        return (per_token * batch["attention_mask"][:, _TOKENS_AFTER_THE_FIRST]).sum(dim=-1)
