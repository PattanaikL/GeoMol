Check raw codes
- [ ] `train.py`
- 由于开启了`num_workers !=0`，所以当第一个error出现时，会使得第一个worker的state为0,继而引发第二个错误，所以最直接有效合理的解决方法是解决第一个error
- 也可能是不支持num_workers != 0，这里使用比较ugly的解决方法，即将num_workers 设置为0
Possible solution:
- Related links:
  - https://blog.csdn.net/qq_29598161/article/details/118444118
  - https://discuss.pytorch.org/t/not-using-multiprocessing-but-getting-cuda-error-re-forked-subprocess/54610


#### Other warnings:
_This warning message exists either at the start and EOF of the output/log file_
> elinks: symbol lookup error: /lib64/libk5crypto.so.3: undefined symbol: EVP_KDF_ctrl, version OPENSSL_1_1_1b

- looks like problem with machine, is a warning



#### Warning on ampere:
> NVIDIA GeForce RTX 3090 with CUDA capability sm_86 is not compatible with the current PyTorch installation.
> The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_61 sm_70 sm_75 compute_37.
> If you want to use the NVIDIA GeForce RTX 3090 GPU with PyTorch, please check the instructions at https://pytorch.org/get-started/locally/
> 
>   warnings.warn(incompatible_device_warn.format(device_name, capability, " ".join(arch_list), device_name))
> /pubhome/qcxia02/miniconda3/envs/GeoMol/lib/python3.7/site-packages/torch/functional.py:1069: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  /opt/conda/conda-bld/pytorch_1634272126608/work/aten/src/ATen/native/TensorShape.cpp:2157.)
>   return _VF.cartesian_prod(tensors)  # type: ignore[attr-defined]

- ampere problem, will increase the computation time

***
- process for dataloader enumeration
```python
for i, data in tqdm(enumerate(loader), total=len(loader)):
# Because of Iteration, __next__ is used
def __next__(self) -> Any:
    with torch.autograd.profiler.record_function(self._profile_name):
        if self._sampler_iter is None:
            self._reset()
        data = self._next_data() # This line #
        self._num_yielded += 1
    return data

def _next_data(self):
    while True:
        idx, data = self._get_data() # where data from
        self._tasks_outstanding -= 1
        if self._dataset_kind == _DatasetKind.Iterable:
            # Check for _IterableDatasetStopIteration
            if isinstance(data, _utils.worker._IterableDatasetStopIteration):
                if self._persistent_workers:
                    self._workers_status[data.worker_id] = False
                else:
                    self._mark_worker_as_unavailable(data.worker_id)
                self._try_put_index()
                continue

            if idx != self._rcvd_idx:
                # store out-of-order samples
                self._task_info[idx] += (data,)
            else:
                del self._task_info[idx]
                return self._process_data(data) # This line #

def _get_data(self):
    success, data = self._try_get_data()
    if success:
        return data

def _try_get_data(self, timeout=_utils.MP_STATUS_CHECK_INTERVAL):
    try:
        data = self._data_queue.get(timeout=timeout) # key for data from
        return (True, data)

def _process_data(self, data):
    self._rcvd_idx += 1
    self._try_put_index()
    if isinstance(data, ExceptionWrapper):
        data.reraise()
    return data
``` 

- content of mol of dataset in one dataloader
```python
train_loader.dataset[0]
shape
> Data(
  x=[15, 44],
  edge_index=[2, 30],
  edge_attr=[30, 4],
  pos=[1],
  z=[15],
  neighbors={
    0=[4],
    4=[4],
    5=[2],
    7=[3],
    8=[3],
    9=[2],
    10=[2],
    11=[3]
  },
  chiral_tag=[15],
  name='C[C@@H](C#N)c1cnoc1',
  boltzmann_weight=0.41841,
  degeneracy=3,
  mol=<rdkit.Chem.rdchem.Mol object at 0x7fffa0929070>,
  pos_mask=[10],
  edge_index_dihedral_pairs=[2, 8]
)

special variables
function variables
batch:None
edge_attr:tensor([[1., 0., 0., 0.],
        [1., 0., 0., 0.],
        [1., 0., 0., 0.],
        [1., 0., 0., 0.],
        [1., 0., 0., 0.],
        [1., 0., 0., 0.],
        [1., 0., 0., 0.],
        [1., 0., 0., 0.],
        [1., 0., 0., 0.],
        [1., 0., 0., 0.],
        [1., 0., 0., 0.],
        [1., 0., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 1., 0.],
        [1., 0., 0., 0.],
        [0., 0., 0., 1.],
        [0., 0., 0., 1.],
        [0., 0., 0., 1.],
        [0., 0., 0., 1.],
        [1., 0., 0., 0.],
        [0., 0., 0., 1.],
        [0., 0., 0., 1.],
        [0., 0., 0., 1.],
        [0., 0., 0., 1.],
        [0., 0., 0., 1.],
        [0., 0., 0., 1.],
        [1., 0., 0., 0.],
        [1., 0., 0., 0.],
        [1., 0., 0., 0.],
        [1., 0., 0., 0.]])
edge_index:tensor([[ 0,  0,  0,  0,  1,  2,  3,  4,  4,  4,  4,  5,  5,  6,  7,  7,  7,  8,
          8,  8,  9,  9, 10, 10, 11, 11, 11, 12, 13, 14],
        [ 1,  2,  3,  4,  0,  0,  0,  0,  5,  7, 14,  4,  6,  5,  4,  8, 11,  7,
          9, 13,  8, 10,  9, 11,  7, 10, 12, 11,  8,  4]])
edge_stores:[{'x': tensor([[0., 1...11,  7]])}]
keys:['degeneracy', 'name', 'edge_index', 'edge_index_dihedral_pairs', 'mol', 'neighbors', 'edge_attr', 'pos_mask', 'pos', 'chiral_tag', 'boltzmann_weight', 'z', 'x']
node_stores:[{'x': tensor([[0., 1...11,  7]])}]
num_edge_features:4
num_edges:30
num_faces:None
```

- training pickle
```python
"O.pickle"
{
'conformers': 
        [{'geom_id': 120780828, 'set': 1, 'degeneracy': 1, 'totalenergy': -5.07054445, 'relativeenergy': 0.0, 'boltzmannweight': 1.0, 'conformerweights': [1.0], 'rd_mol': <rdkit.Chem.rdchem.Mol object at 0x7f9749d035b0>}]
'totalconfs': 1
'temperature': 298.15
'uniqueconfs': 1
'lowestenergy': -5.07054
'poplowestpct': 100.0
'ensembleenergy': 0.0
'ensembleentropy': 0.0
'ensemblefreeenergy': 0.0
'charge': 0
'smiles': 'O'
}
```