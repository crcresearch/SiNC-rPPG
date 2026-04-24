"""Central registration of dataset name strings to dataset classes."""

from __future__ import annotations

from utils.registry import Registry

DATASET_REGISTRY = Registry()


def _register_datasets() -> None:
    from datasets.PURE_supervised import PURESupervised as PURESupervisedTrain
    from datasets.PURE_testing import PURESupervised as PURESupervisedTest
    from datasets.PURE_unsupervised import PUREUnsupervised
    from datasets.UBFC_supervised import UBFCSupervised as UBFCSupervisedTrain
    from datasets.UBFC_testing import UBFCSupervised as UBFCSupervisedTest
    from datasets.UBFC_unsupervised import UBFCUnsupervised

    DATASET_REGISTRY.register("pure_unsupervised", PUREUnsupervised)
    DATASET_REGISTRY.register("pure_supervised", PURESupervisedTrain)
    DATASET_REGISTRY.register("pure_testing", PURESupervisedTest)
    DATASET_REGISTRY.register("ubfc_unsupervised", UBFCUnsupervised)
    DATASET_REGISTRY.register("ubfc_supervised", UBFCSupervisedTrain)
    DATASET_REGISTRY.register("ubfc_testing", UBFCSupervisedTest)


_register_datasets()
