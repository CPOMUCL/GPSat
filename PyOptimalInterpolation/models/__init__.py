from PyOptimalInterpolation.models.base_model import BaseGPRModel
from PyOptimalInterpolation.models.gpflow_models import GPflowGPRModel, GPflowSGPRModel
try:
    from PyOptimalInterpolation.models.sklearn_models import sklearnGPRModel
    from PyOptimalInterpolation.models.vff_model import GPflowVFFModel
except:
    pass