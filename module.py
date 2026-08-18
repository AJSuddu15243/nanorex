from __future__ import annotations
from typing import Dict, Iterator, Tuple
import numpy as np

class Module:

  def __init__(self):
    self.params: Dict[str, np.darray] = {}
    self.grads: Dict[str, np.ndarray] = {}
    self._children: Dict[str, "Module"] = {}

  def register(self, name: str, child: "Module"):
    self._children[name] = child
    return child

  def param_items(self, prefix: str = ""):
    for k, p in self.params.items():
      yield prefix + k, p , self.grads[k]
    for name, child in self._children.items():
      yield from child.param_items(prefix + name + ".")

  def num_params(self):
    return int(sum(p.size for _, p, _ in self.param.items()))