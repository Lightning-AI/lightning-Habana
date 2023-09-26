# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2023-09-26


### Added

- Documentation with examples for using DeepSpeed with HPU ([#64](https://github.com/Lightning-AI/lightning-Habana/pull/64))

- Add autocast using HPUPrecision plugin ([#66](https://github.com/Lightning-AI/lightning-Habana/pull/66), [#75](https://github.com/Lightning-AI/lightning-Habana/pull/75))

- Demonstrate HPU Graphs support ([#67](https://github.com/Lightning-AI/lightning-Habana/pull/67))

- Enhance test coverage of DeepSpeed strategy on HPU ([#68](https://github.com/Lightning-AI/lightning-Habana/pull/68))

- Added version check helper to use right release ([#75](https://github.com/Lightning-AI/lightning-Habana/pull/75), [#76](https://github.com/Lightning-AI/lightning-Habana/pull/76))

- Implement reduce with parallel plugin ([#77](https://github.com/Lightning-AI/lightning-Habana/pull/77))

### Changed

- Changed HPU docker image based on synapse AI release 1.11.0 & upgraded deepspeed plugin to version 0.9.4 ([#61](https://github.com/Lightning-AI/lightning-Habana/pull/61))


### Fixed

- Fixed optimizer priority based on deepspeed specification ([#36](https://github.com/Lightning-AI/lightning-Habana/pull/69))

- Fixed missing extras in package ([#70](https://github.com/Lightning-AI/lightning-Habana/pull/70))

-


### Removed

-


### Deprecated

- Warn on HMP deprecation from `HPUPrecision` plugin ([#65](https://github.com/Lightning-AI/lightning-Habana/pull/65))


## [1.0.1] - 2023-07-26


### Added

- Added tests for mixed precision training ([#36](https://github.com/Lightning-AI/lightning-Habana/pull/36))
- Example to include mixed precision training ([#54](https://github.com/Lightning-AI/lightning-Habana/pull/54))

### Changed

- Enabled skipped tests based on registered strategy, accelerator ([#46](https://github.com/Lightning-AI/lightning-Habana/pull/46))

### Fixed

- Fixed Attribute Error ([#43](https://github.com/Lightning-AI/lightning-Habana/pull/43))
- Fixed wrong imports ([#44](https://github.com/Lightning-AI/lightning-Habana/pull/44))
- Fixed graph breaks in test/val phases in lazy mode ([#45](https://github.com/Lightning-AI/lightning-Habana/pull/45))


## [1.0.0] - 2023-06-14

### Added

- Added HPU support for fabric ([#11](https://github.com/Lightning-AI/lightning-Habana/pull/11))
- Added Pytorch HPU profiler support ([#15](https://github.com/Lightning-AI/lightning-Habana/pull/15))
- Added basic HPU infra support for deep speed ([#21](https://github.com/Lightning-AI/lightning-Habana/pull/21))
- Added Pytorch HPU datamodule support ([#16](https://github.com/Lightning-AI/lightning-Habana/pull/16))

### Changed

- Changed code hierarchy in compliance with base lightning code for pytorch ([#12](https://github.com/Lightning-AI/lightning-Habana/pull/12))
- Changed default HPU docker image based on HPU release 1.10.0 ([#30](https://github.com/Lightning-AI/lightning-Habana/pull/30))
-
### Fixed

- Fixed mnist example test ([#20](https://github.com/Lightning-AI/lightning-Habana/pull/20))
- Habana dataloader hang with Lightning 2.0.x ([#29](https://github.com/Lightning-AI/lightning-Habana/pull/29))
- Make #29 applicable only for gaudi devices ([#39](https://github.com/Lightning-AI/lightning-Habana/pull/39))
- Fixed environment initialization for hpus and fixed docs ([#40](https://github.com/Lightning-AI/lightning-Habana/pull/40))
- Fixed docs and added work around to make use hpu media packages without signature issues ([#41](https://github.com/Lightning-AI/lightning-Habana/pull/41))

### Removed

- Cleaning up env's ID for HPU parallel plugins based on synapse AI release 1.9 ([#28](https://github.com/Lightning-AI/lightning-Habana/pull/28))
- Remove unnecessary import checks which degrade performance ([#38](https://github.com/Lightning-AI/lightning-Habana/pull/38))

### Deprecated
