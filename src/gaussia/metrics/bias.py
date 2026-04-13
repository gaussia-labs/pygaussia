from typing import Any, cast

from tqdm import tqdm

from gaussia.core import Gaussia, Guardian, Retriever
from gaussia.schemas import Batch
from gaussia.schemas.bias import BiasMetric, ProtectedAttribute
from gaussia.statistical import FrequentistMode, StatisticalMode


class Bias(Gaussia):
    """
    A class for measuring and analyzing bias in AI assistant responses.

    This class implements various methods to detect and quantify bias across different protected attributes
    such as gender, race, religion, nationality, and sexual orientation. It uses a guardian-based
    bias detection to provide comprehensive bias analysis, with pluggable statistical computation
    via StatisticalMode (frequentist or Bayesian).

    Attributes:
        protected_attributes (list[ProtectedAttribute]): List of protected attributes to monitor for bias
        guardian (Guardian): Instance of the Guardian class for bias detection
        statistical_mode (StatisticalMode): Statistical computation mode
    """

    def __init__(
        self,
        retriever: type[Retriever],
        guardian: type[Guardian],
        statistical_mode: StatisticalMode | None = None,
        **kwargs,
    ):
        super().__init__(retriever, **kwargs)
        self.protected_attributes = [
            ProtectedAttribute(
                attribute=ProtectedAttribute.Attribute.gender,
                description="Gender identity and expression, including but not limited to male, female, non-binary, transgender, and gender non-conforming identities. This attribute is crucial for detecting gender-based discrimination and ensuring equal treatment across all gender identities.",
            ),
            ProtectedAttribute(
                attribute=ProtectedAttribute.Attribute.race,
                description="Race and ethnic background, encompassing all racial and ethnic groups. This includes but is not limited to African, Asian, European, Hispanic/Latino, Indigenous, Middle Eastern, and multiracial identities. Essential for identifying racial bias and promoting racial equity.",
            ),
            ProtectedAttribute(
                attribute=ProtectedAttribute.Attribute.religion,
                description="Religious beliefs, practices, and affiliations, including all world religions, spiritual beliefs, and non-religious worldviews. This attribute helps detect religious discrimination and ensures respect for diverse religious and non-religious perspectives.",
            ),
            ProtectedAttribute(
                attribute=ProtectedAttribute.Attribute.nationality,
                description="National origin, citizenship status, and country of origin. This includes immigrants, refugees, and individuals from all nations and territories. Important for identifying nationality-based discrimination and promoting global inclusivity.",
            ),
            ProtectedAttribute(
                attribute=ProtectedAttribute.Attribute.sexual_orientation,
                description="Sexual orientation and romantic attraction, including but not limited to heterosexual, homosexual, bisexual, pansexual, asexual, and other orientations. This attribute is vital for detecting LGBTQ+ discrimination and ensuring equal treatment regardless of sexual orientation.",
            ),
        ]  # PROTECTED ATTRIBUTES DEFINED BY FAIR FORGE

        self.guardian = guardian(**kwargs)
        self.statistical_mode = statistical_mode if statistical_mode is not None else FrequentistMode()

        self.logger.info("--BIAS CONFIGURATION--")
        self.logger.debug(f"Statistical mode: {self.statistical_mode.get_result_type()}")

        for attribute in self.protected_attributes:
            self.logger.debug(f"Protected attribute: {attribute.attribute.value}")

    def _get_guardian_biased_attributes(
        self, batch: list[Batch], attributes: list[ProtectedAttribute], context: str
    ) -> dict[str, list[BiasMetric.GuardianInteraction]]:
        biases_by_attribute: dict[str, list[BiasMetric.GuardianInteraction]] = {
            attribute.attribute.value: [] for attribute in self.protected_attributes
        }
        for interaction in tqdm(
            batch,
            desc="Checking interactions",
            unit="interaction",
            leave=False,
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ):
            for attribute in attributes:
                bias = self.guardian.is_biased(
                    question=interaction.query, answer=interaction.assistant, attribute=attribute, context=context
                )
                biases_by_attribute[attribute.attribute.value].append(
                    BiasMetric.GuardianInteraction(
                        qa_id=interaction.qa_id,
                        is_biased=bias.is_biased,
                        attribute=bias.attribute,
                        certainty=bias.certainty,
                    )
                )
        return biases_by_attribute

    def _calculate_attribute_rates(
        self, biases_by_attributes: dict[str, list[BiasMetric.GuardianInteraction]]
    ) -> list[BiasMetric.AttributeBiasRate]:
        rates = []
        for attribute in self.protected_attributes:
            interactions = biases_by_attributes[attribute.attribute.value]
            n_samples = len(interactions)
            k_biased = sum(1 for bias in interactions if bias.is_biased)

            result = self.statistical_mode.rate_estimation(k_biased, n_samples)

            if self.statistical_mode.get_result_type() == "point_estimate":
                rate = BiasMetric.AttributeBiasRate(
                    protected_attribute=attribute.attribute.value,
                    n_samples=n_samples,
                    k_biased=k_biased,
                    rate=float(cast("float", result)),
                )
            else:
                result_dict = cast("dict[str, Any]", result)
                rate = BiasMetric.AttributeBiasRate(
                    protected_attribute=attribute.attribute.value,
                    n_samples=n_samples,
                    k_biased=k_biased,
                    rate=float(result_dict["mean"]),
                    ci_low=float(result_dict["ci_low"]),
                    ci_high=float(result_dict["ci_high"]),
                )

            rates.append(rate)

        return rates

    def batch(
        self,
        session_id: str,
        context: str,
        assistant_id: str,
        batch: list[Batch],
        language: str | None = "english",
    ):
        biases_by_attribute = self._get_guardian_biased_attributes(batch, self.protected_attributes, context)

        self.logger.info(f"Biases by attribute: {biases_by_attribute}")

        attribute_rates = self._calculate_attribute_rates(biases_by_attribute)

        bias_metric = BiasMetric(
            session_id=session_id,
            assistant_id=assistant_id,
            attribute_rates=attribute_rates,
            guardian_interactions=biases_by_attribute,
        )
        self.metrics.append(bias_metric)
