#!/bin/bash
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

# THIS FILE ASSUMES IT IS RUN INSIDE THE tests DIRECTORY
set -e

# Defaults
hpus=2

# Parse input args
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --hpus)
            hpus="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Get all the tests marked with standalone marker
TEST_FILE="standalone_tests.txt"
python -um pytest tests/test_pytorch tests/test_fabric -q --collect-only -m standalone --pythonwarnings ignore > $TEST_FILE
cat $TEST_FILE
sed -i '$d' $TEST_FILE

# Declare an array to store test results
declare -a results

# Get test list and run each test individually
tests=$(grep -oP '^tests/test_\S+' "$TEST_FILE")
for test in $tests; do
  result=$(python -um pytest -sv "$test" --hpus $hpus --pythonwarnings ignore --junitxml="$test"-results.xml | tail -n 1)
  pattern='([0-9]+) (.*) in ([0-9.]+s)'
  status=""
  if [[ $result =~ $pattern ]]; then
      status="${BASH_REMATCH[2]}"
  fi
  result="$test:${status^^}"
  echo $result
  if [[ $status == "failed" ]]; then
    cat $test-results.xml
    exit 1
  fi
  results+=("$result")
done

echo "===== STANDALONE TEST STATUS BEGIN ====="
for result in "${results[@]}"; do
  echo "$result"
done
echo "===== STANDALONE TEST STATUS END ====="

mv test_pytorch/*.xml .
rm $TEST_FILE
