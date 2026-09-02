from importlib import import_module


MODEL_MODULES = {
    "BERT4Rec": "BERT4Rec",
    "BSARec": "BSARec",
    "CL4SRec": "CL4SRec",
    "CoSeRec": "CoSeRec",
    "DuoRec": "DuoRec",
    "FMLPRec": "FMLPRec",
    "GRU4Rec": "GRU4Rec",
    "ICLRec": "ICLRec",
    "ICSRec": "ICSRec",
    "IDURL": "IDURL",
    "IOCRec": "IOCRec",
    "SASRec": "SASRec",
    "WEARec": "WEARec",
}


def get_model_class(model_name):
    module_name = MODEL_MODULES.get(model_name)
    if module_name is None:
        supported = ", ".join(sorted(MODEL_MODULES))
        raise ValueError(
            f"Model {model_name} is not supported. Supported models: {supported}."
        )
    module = import_module(f"{__name__}.{module_name}")
    return getattr(module, model_name)
