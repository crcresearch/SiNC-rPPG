import sys

from datasets.dataset_registry import DATASET_REGISTRY


def get_dataset(split, arg_obj):
    key = arg_obj.dataset.lower()
    try:
        cls = DATASET_REGISTRY.get(key)
    except KeyError:
        print("Dataset not found:", key, "Exiting.")
        sys.exit(-1)
    print(f"Using dataset class {cls.__name__} ({key}).")
    return cls(split, arg_obj)
