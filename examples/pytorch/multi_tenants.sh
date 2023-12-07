# Copyright The Lightning AI team.
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


#!/bin/bash

HABANA_VISIBLE_MODULES="0,1" MASTER_PORT=1234 python -u  mnist_trainer.py -v --run_type="basic" --devices="2" &
HABANA_VISIBLE_MODULES="2,3" MASTER_PORT=1244 python -u  mnist_trainer.py -v --run_type="basic" --devices="2" &
HABANA_VISIBLE_MODULES="4,5" MASTER_PORT=1255 python -u  mnist_trainer.py -v --run_type="basic" --devices="2" &
HABANA_VISIBLE_MODULES="6,7" MASTER_PORT=1266 python -u  mnist_trainer.py -v --run_type="basic" --devices="2" &

wait
