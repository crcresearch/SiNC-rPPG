import sys

from utils.model_registry import MODEL_REGISTRY


def select_model(arg_obj):
    model_type = arg_obj.model_type.lower()
    channels = arg_obj.channels
    input_channels = len(channels)
    dropout = float(arg_obj.dropout)

    try:
        cls = MODEL_REGISTRY.get(model_type)
    except KeyError:
        print("Could not find model specified:", model_type)
        sys.exit(-1)

    return cls(input_channels=input_channels, drop_p=dropout)
