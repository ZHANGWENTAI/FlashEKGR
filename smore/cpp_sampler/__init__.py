# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

# Try to import libsampler from build/dll directory
# First check if it's already in sys.path (e.g., via PYTHONPATH)
try:
    import libsampler
except ImportError:
    # If not found, add build/dll to sys.path
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dll_path = os.path.join(dir_path, "build/dll")
    if dll_path not in sys.path:
        sys.path.insert(0, dll_path)
    import libsampler
