from GPSat.models.base_model import BaseGPRModel

def get_model(name):
    # IMPORT MODELS THROUGH THIS FUNCTION
    model = None

    if name == "GPflowGPRModel":
        from GPSat.models.gpflow_models import GPflowGPRModel as model
    elif name == 'GPflowSGPRModel':
        from GPSat.models.gpflow_models import GPflowSGPRModel as model
    elif name == 'GPflowSVGPModel':
        from GPSat.models.gpflow_models import GPflowSVGPModel as model
    elif name == "sklearnGPRModel":
        from GPSat.models.sklearn_models import sklearnGPRModel as model
    elif name == "GPflowVFFModel":
        from GPSat.models.vff_model import GPflowVFFModel as model
    elif name == "GPflowASVGPModel":
        from GPSat.models.asvgp_model import GPflowASVGPModel as model
    elif name == "PurePythonGPR":
        from GPSat.models.pure_python_gpr import PurePythonGPR as model
    elif name == "GPyTorchGPRModel":
        from GPSat.models.gpytorch_models import GPyTorchGPRModel as model
    else:
        raise NotImplementedError(f"model with name: '{name}' is not implemented")

    assert model is not None

    return model
