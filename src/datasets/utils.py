import sys

from torch.utils.data import DataLoader, WeightedRandomSampler

from datasets.dataset_registry import DATASET_REGISTRY


def build_training_dataloader(train_set, arg_obj):
    """Shuffle uniformly over concat, or ``WeightedRandomSampler`` when ``train_set.sample_weights`` is set."""
    sw = getattr(train_set, "sample_weights", None)
    if sw is not None:
        sampler = WeightedRandomSampler(
            weights=sw,
            num_samples=len(train_set),
            replacement=True,
        )
        return DataLoader(
            train_set,
            batch_size=int(arg_obj.batch_size),
            shuffle=False,
            sampler=sampler,
            num_workers=int(arg_obj.num_workers),
        )
    return DataLoader(
        train_set,
        batch_size=int(arg_obj.batch_size),
        shuffle=True,
        num_workers=int(arg_obj.num_workers),
    )


def get_dataset(split, arg_obj):
    key = arg_obj.dataset.lower()
    try:
        cls = DATASET_REGISTRY.get(key)
    except KeyError:
        print("Dataset not found:", key, "Exiting.")
        sys.exit(-1)
    print(f"Using dataset class {cls.__name__} ({key}).")
    return cls(split, arg_obj)
