# Copyright (c) 2023 Habana Labs, Ltd. an Intel Company
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

def get_version():
    version = os.getenv('RELEASE_VERSION')
    if not version:
        version = "0.0.0"
    build_number = os.getenv('RELEASE_BUILD_NUMBER')
    if build_number:
        return version + '.' + build_number
    else:
        try:
            import subprocess
            root = os.environ["PYTORCH_MODULES_ROOT_PATH"]
            sha = (
                subprocess.check_output(
                    ["git", "-C", root, "rev-parse", "--short", "HEAD"])
                .decode("ascii").strip())
            return f"{version}+git{sha}"
        except Exception as e:
            print("Error getting version: {}".format(e), file=sys.stderr)
            return f"{version}+unknown"

__version__ = get_version()
__author__ = "Lightning-AI et al."
__author_email__ = "name@lightning.ai"
__license__ = "Apache-2.0"
__copyright__ = f"Copyright (c) 2020-2023, {__author__}."
__homepage__ = "https://github.com/Lightning-AI/lightning-habana"
__docs__ = "Lightning suport for Intel Habana accelerators"

__all__ = [
    "__author__",
    "__author_email__",
    "__copyright__",
    "__docs__",
    "__homepage__",
    "__license__",
    "__version__",
]
