# from .jasper_ridge import JasperRidgeDataset, MotionCodeJasperRidge
# from .urban import UrbanDataset, MotionCodeUrban
from .jasper_ridge import JasperRidgeDataset
from .urban import UrbanDataset

dataset_factory = {
    'jasper_ridge': JasperRidgeDataset,
    # 'jasper_ridge_pixel': MotionCodeJasperRidge,
    'urban': UrbanDataset,
    # 'urban_pixel': MotionCodeUrban
}
