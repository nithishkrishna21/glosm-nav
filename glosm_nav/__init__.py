# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

# Backwards-compat: this package was renamed "vlfm" -> "glosm_nav". Pretrained
# artifacts from the original VLFM release (e.g. data/pointnav_weights.pth) have the
# old "vlfm.*" module paths pickled inside them. Redirect any "vlfm.*" import to the
# identical, already-loaded "glosm_nav.*" module so those pickles unpickle without
# re-importing (which would re-run habitat's obs-transform/policy registrations).
import importlib
import importlib.abc
import importlib.util
import sys

_OLD_NAME = "vlfm"
_NEW_NAME = __name__


class _VlfmAliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path=None, target=None):
        if name == _OLD_NAME or name.startswith(_OLD_NAME + "."):
            return importlib.util.spec_from_loader(name, self)
        return None

    def create_module(self, spec):
        module = importlib.import_module(_NEW_NAME + spec.name[len(_OLD_NAME):])
        sys.modules[spec.name] = module
        return module

    def exec_module(self, module):
        pass


sys.modules.setdefault(_OLD_NAME, sys.modules[_NEW_NAME])
if not any(isinstance(f, _VlfmAliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _VlfmAliasFinder())
