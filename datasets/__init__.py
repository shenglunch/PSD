from .data_nyu import NYUDataset
from .data_diode_in import DIODEiDataset
from .data_sunrgbd import SUNRGBDDataset
from .data_scannet import SCANNETDataset
from .data_middlebury import MIDDLEBURYDataset
from .data_hypersim import HypersimDataset
from .data_eth3d import Eth3DDataset
from .data_void import VOIDDataset
from .data_kitti import KittiDataset
from .data_drivingstereo import DrivingStereoDataset
from .data_diode_out import DIODEoDataset
from .data_argoverse import ARGOVERSEDataset
from .data_vkitti2 import VKITTI2Dataset
from .data_diml_in import DIMLiDataset
from .data_cityscape import CityscapeDataset
from .data_tofdc import TOFDCDataset
from .data_hammer import HAMMERDataset
from .data_stanford import StanfordDataset
from .data_kitti360 import KITTI360Dataset
from .data_nk import NKDataset

__datasets__ = {
    "NYUv2": NYUDataset,
    "DIODEi": DIODEiDataset,
    "SUNRGBD": SUNRGBDDataset,
    "SCANNET": SCANNETDataset,
    "MIDDLEBURY": MIDDLEBURYDataset,
    "HYPERSIM": HypersimDataset,
    "ETH3D": Eth3DDataset,
    "VOID1500": VOIDDataset,
    "KITTI": KittiDataset,
    "DrivingStereo": DrivingStereoDataset,
    "DIODEo": DIODEoDataset,
    "ARGOVERSE": ARGOVERSEDataset,
    "VKITTI2": VKITTI2Dataset,
    "DIMLi": DIMLiDataset,
    "Cityscape": CityscapeDataset,
    "TOFDC": TOFDCDataset,
    "HAMMER": HAMMERDataset,
    "Stanford": StanfordDataset,
    "KITTI360": KITTI360Dataset,
    "NK": NKDataset}
