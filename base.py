from abc import ABC, abstractmethod

class LPModelWrapper(ABC):
    """模型基类:定义通用接口"""
    @abstractmethod
    def load_layer_tensor(self, layer_idx, path):
        pass
        
    @abstractmethod
    def load_weight2layer(self, layer, layer_idx, weight_tensor_layer, non_blocking=True):
        pass
    
    # get init layer
    @abstractmethod
    def get_layer(config, layer_idx: int):
        pass
    
class DecoderLayerWrapper(ABC):
    """模型层:定义通用接口"""
    @abstractmethod
    def get_layer(config, layer_idx: int):
        pass
    