from dataclasses import dataclass
import torch
from typing import List, Optional, Tuple, Union, Dict

@dataclass
class IO_Task:
    type: int
    q_data: Optional[torch.Tensor]
    k_data: Optional[torch.Tensor]
    v_data: Optional[torch.Tensor]
    sin: Optional[torch.Tensor]
    cos: Optional[torch.Tensor]
    
    attn_output: Optional[torch.Tensor]
    
    layer: Optional[Dict]
    
class QKV:
    def __init__(q, k, v, sin, cos, attn_mask):
        self.q = q
        self.k = k
        self.v = v
        self.sin = sin
        self.cos =  cos
        self.attn_mask=attn_mask
    def getq():
        return self.q
    def getk():
        return self.k
    def getv():
        return self.v
    def getsin():
        return self.sin
    def getcos():
        return self.cos
    def getmask():
        return self.attn_mask
    
class GPU_Task:
    def __init__(attn_output, layer):
        self.attn_output=attn_output
        self.layer=layer
    def getOutput():
        return self.attn_output
    def getLayer():
        return self.layer

class Attn_Task:
    def __init__(input, layer):
        self.layer=layer
        self.input=input
    def getLayer():
        return self.layer
    def getInput():
        return self.input