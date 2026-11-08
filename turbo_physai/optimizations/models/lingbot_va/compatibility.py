import hashlib
import importlib
import sys
from pathlib import Path
from turbo_physai.engine.contracts import CompatibilityResult

BASELINE_COMMIT = '7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb'
MODEL_SHA256 = 'f65fdca5f28b327bfc89ca082a7b55eee3fb4ab1ffc469c642a075920e6f4487'

def baseline(context):
    model = importlib.import_module('modules.model')
    other = sys.modules.get('wan_va.modules.model')
    identity_ok = other is None or other.WanTransformer3DModel is model.WanTransformer3DModel
    actual = hashlib.sha256(Path(model.__file__).read_bytes()).hexdigest()
    root = Path(model.__file__).resolve().parents[1]
    extra = {'train.py': '103e04031b32cbbef0c829a7b3f52986aaaffa996040c173ff8f9a67cc4985ab', 'distributed/fsdp.py': 'cf5c7755e934f933645235f9104cae3ff9adea2b2863bcb1a5ec42eb6f3a8d8f'}
    extra_ok = all(hashlib.sha256((root/name).read_bytes()).hexdigest() == value for name, value in extra.items())
    return CompatibilityResult(actual == MODEL_SHA256 and identity_ok and extra_ok,
        expected={'sha256': MODEL_SHA256, 'single_module_identity': True},
        actual={'sha256': actual, 'single_module_identity': identity_ok},
        reason='LingBot-VA fixed source and canonical modules.model identity required')
