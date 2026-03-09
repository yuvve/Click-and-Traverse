from dataclasses import dataclass
from cyclonedds.idl import IdlStruct
import cyclonedds.idl as idl

import numpy as np
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types


@dataclass
@annotate.final
@annotate.autoid("sequential")
class VoxelMap(IdlStruct, typename="octomap_idl::msg::VoxelMap"):
    length: int = 0
    width: int = 0
    height: int = 0
    data_sdf: str = ""
    data_bf: str = ""
    resolution: float = 0
