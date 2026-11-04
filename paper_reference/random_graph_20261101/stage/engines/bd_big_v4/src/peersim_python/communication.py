"""Shared, algorithm-independent gossip schedule and numeric wire-size policy."""
import hashlib,random,numbers
from collections.abc import Mapping

POLICY='numeric-v1: array dtype bytes; numeric scalars/IDs 8 bytes; fixed schema keys and physical headers excluded'


def event_rng(seed,cycle,node_id,stream):
    digest=hashlib.sha256(f'{seed}:{cycle}:{node_id}:{stream}'.encode('ascii')).digest()
    return random.Random(int.from_bytes(digest[:16],'big'))


def numeric_payload_nbytes(value):
    # Torch and NumPy arrays preserve their real numeric precision.
    if hasattr(value,'numel') and hasattr(value,'element_size'):
        return int(value.numel()*value.element_size())
    if hasattr(value,'nbytes') and hasattr(value,'shape') and len(value.shape)>0:
        return int(value.nbytes)
    if value is None:return 0
    if isinstance(value,numbers.Number):return 8
    if isinstance(value,Mapping):
        return sum((8 if isinstance(k,numbers.Number) else 0)+numeric_payload_nbytes(v) for k,v in value.items())
    if isinstance(value,(tuple,list)):return sum(numeric_payload_nbytes(v) for v in value)
    if isinstance(value,str):return 0  # keys/enum tags are predefined schema
    raise TypeError(f'Unsupported wire payload value: {type(value).__name__}')


def activation_order(size,seed,cycle,activation):
    rng=event_rng(seed,cycle,-1,'activation')
    if activation=='shuffle':
        order=list(range(size));rng.shuffle(order);return order
    if activation=='getpair_rand':return [rng.randrange(size) for _ in range(size)]
    return list(range(size))


def selected_neighbors(node,seed,cycle,k,pid=0):
    link=node.getProtocol(pid)
    neighbors=sorted([link.getNeighbor(j) for j in range(link.degree())],key=lambda n:n.getIndex())
    return event_rng(seed,cycle,node.getIndex(),'neighbors').sample(neighbors,min(k,len(neighbors)))
