from ._base import EmissionModel
from petgem.emission.types import EmissionTypes


class BackToBack(EmissionModel):
    type = EmissionTypes.BACKTOBACK