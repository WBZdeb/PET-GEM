from _base import EmissionModel
from petgem.emission.types import EmissionTypes


class SingleGamma(EmissionModel):
    mode = EmissionTypes.BACKTOBACK