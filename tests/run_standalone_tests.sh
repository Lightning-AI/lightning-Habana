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
RED='\033[0;31m'
NC='\033[0m'

# Defaults
hpus=2
files=""
filter=""

# Parse input args
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --hpus)
            hpus="$2"
            shift 2
            ;;
        -f|--files)
            shift
            while [[ "$1" != -* && ! -z "$1" ]]; do
              files+=" $1"
              shift
            done
            ;;
        -k|--filter)
            filter="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$files" ]]; then
  test_files="tests/test_pytorch tests/test_fabric"
else
  test_files=$files
fi
echo "Test files: $test_files"

# Get all the tests marked with standalone marker
TEST_FILE="standalone_tests.txt"
test_command="python -um pytest $test_files -q --collect-only -m standalone --pythonwarnings ignore"
if [[ -n "$filter" ]]; then
  test_command+=" -k $filter"
fi

$test_command > $TEST_FILE
cat $TEST_FILE
sed -i '$d' $TEST_FILE

# Declare an array to store test results
declare -a results

# Get test list and run each test individually
tests=$(grep -oP '^tests/test_\S+' "$TEST_FILE")
for test in $tests; do
  result=$(python -um pytest -sv "$test" --hpus $hpus --pythonwarnings ignore --junitxml="$test"-results.xml)
  retval=$?
  last_line=$(tail -n 1 <<< "$result")

  pattern='([0-9]+) (.*) in ([0-9.]+s)'
  status=""
  if [[ $last_line =~ $pattern ]]; then
      status="${BASH_REMATCH[2]}"
  elif [ "$retval" != 0 ]; then
    echo -e "${RED}Got status ${retval} from pytest${NC}"
    echo -e "${RED}$(cat $test-results.xml)${NC}"
    exit 1
  fi

  results+=("${test}:${status}")
done

echo "===== STANDALONE TEST STATUS BEGIN ====="
for result in "${results[@]}"; do
  echo $result
done
echo "===== STANDALONE TEST STATUS END ====="

mv tests/**/*.xml .
rm $TEST_FILE
