# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.7.0] - 2024-10-DD

### Added

- Added utility to get device name from hl-smi ([#232](https://github.com/Lightning-AI/lightning-Habana/pull/232))
- Integrated Intel Neural Compressor for FP8 inference ([#235](https://github.com/Lightning-AI/lightning-Habana/pull/235))
-

### Changed

- Updated to Intel Gaudi software Release 1.16.2 ([#207](https://github.com/Lightning-AI/lightning-Habana/pull/207))
- Updated to Intel Gaudi software Release 1.17.0 ([#221](https://github.com/Lightning-AI/lightning-Habana/pull/221))
- Modified torch device specification for FSDP on HPU ([#222](https://github.com/Lightning-AI/lightning-Habana/pull/222))
- Updated strategy to use default fork ([#234](https://github.com/Lightning-AI/lightning-Habana/pull/234))
- Updated hpu parallel strategy as base class ([#237](https://github.com/Lightning-AI/lightning-Habana/pull/237))
- Updated to Intel Gaudi software Release 1.18.0 ([#245](https://github.com/Lightning-AI/lightning-Habana/pull/245))

### Fixed

- Fixed device name retrieval without hlsmi  ([#240](https://github.com/Lightning-AI/lightning-Habana/pull/240))

### Removed

-

### Deprecated

- Deprecated support for Habana Quantization Toolkit. ([#235](https://github.com/Lightning-AI/lightning-Habana/pull/235))
-

## [1.6.0] - 2024-06-28

### Added

- Added support for additional dtypes ([#194](https://github.com/Lightning-AI/lightning-Habana/pull/194))
- Added more tests of FSDP with HPU ([#197](https://github.com/Lightning-AI/lightning-Habana/pull/197))
- Added FSDP strategy with fabric on HPU ([#198](https://github.com/Lightning-AI/lightning-Habana/pull/198))

### Changed

- Updated to common `hpu_backend` interface for compile support. ([#183](https://github.com/Lightning-AI/lightning-Habana/pull/183))
- Updated to Intel Gaudi software Release 1.16.0 ([#191](https://github.com/Lightning-AI/lightning-Habana/pull/191))
- Updated HQT APIs to be in accordance with Intel Gaudi software Release 1.16.0 ([#192](https://github.com/Lightning-AI/lightning-Habana/pull/192))
- Updated HPUPrecisionPlugin for fp8 based on Intel Gaudi software Release 1.16.0. ([#195](https://github.com/Lightning-AI/lightning-Habana/pull/195))

### Fixed

- Fixed deepspeed documentation & tests based on synapse AI release 1.15.1 and latest PTL fabric. ([#184](https://github.com/Lightning-AI/lightning-Habana/pull/184))
- Workaround to resolve label name issue in HPUProfiler with torch.compile. ([#185](https://github.com/Lightning-AI/lightning-Habana/pull/185))
- Fixed incompatibility issue for PyTorch>=2.3.0 ([#193](https://github.com/Lightning-AI/lightning-Habana/pull/193))
-
### Removed

-

### Deprecated

-


## [1.5.0] - 2024-05-03

### Added

- Added support for Intel Gaudi Profiler. Deprecate `HABANA_PROFILE` environment variable from HPUProfiler. ([#158](https://github.com/Lightning-AI/lightning-Habana/pull/158))
- Added support for FP8 inference. ([#162](https://github.com/Lightning-AI/lightning-Habana/pull/162))
- Added support for LightningCLI. ([#173](https://github.com/Lightning-AI/lightning-Habana/pull/173))
- Added experimental support for FSDP on HPU. ([#174](https://github.com/Lightning-AI/lightning-Habana/pull/174))
- Added support for FP8 inference with DeepSpeed. ([#176](https://github.com/Lightning-AI/lightning-Habana/pull/176))
- Updated the lightning version check for using FSDP. ([#182](https://github.com/Lightning-AI/lightning-Habana/pull/182))


### Changed

- Changed HPUParallelStrategy to HPUDDPStrategy ([#160](https://github.com/Lightning-AI/lightning-Habana/pull/160))
- Changed HPU docker image based on Synapse AI release 1.15.0 ([#166](https://github.com/Lightning-AI/lightning-Habana/pull/166))
- Updated to Intel Gaudi software Release 1.15.1 ([#171](https://github.com/Lightning-AI/lightning-Habana/pull/171))

### Fixed

- Fixed "No profiler activity found" error with HPUProfiler. ([#172](https://github.com/Lightning-AI/lightning-Habana/pull/172))

### Removed

-

### Deprecated

-


## [1.4.0] - 2024-02-16

### Added

- Added DeepSpeed precision plugin for HPU ([#147](https://github.com/Lightning-AI/lightning-Habana/pull/147))
- Added support for fp8 training. ([#149](https://github.com/Lightning-AI/lightning-Habana/pull/149))

### Changed

- Decoupled return strings of firmware, synapse version helper ([#137](https://github.com/Lightning-AI/lightning-Habana/pull/137))
- Changed HPU docker image based on Synapse AI release 1.14.0 ([#140](https://github.com/Lightning-AI/lightning-Habana/pull/140))

### Fixed

- Fixed fabric imports for HPU strategies ([#126](https://github.com/Lightning-AI/lightning-Habana/pull/126))
- Enabling tests and examples of fabric with HPU ([#139](https://github.com/Lightning-AI/lightning-Habana/pull/139))
- Fixes an API break due to non-strict loading in Trainer ([#150](https://github.com/Lightning-AI/lightning-Habana/pull/150))

### Removed

-

### Deprecated

- `aot_hpu_training_backend` will be deprecated. Use `hpu_backend` instead for torch compile with hpu ([#148](https://github.com/Lightning-AI/lightning-Habana/pull/148))


## [1.3.0] - 2023-12-06

### Added

- Added support for Deepspeed inference on HPU with tests and documentation ([#110](https://github.com/Lightning-AI/lightning-Habana/pull/110))
- Added tests, examples, and documentation for dynamic shapes with recipe caching ([#107](https://github.com/Lightning-AI/lightning-Habana/pull/107))
- Added preview of torch compile with tests and documentation ([#119](https://github.com/Lightning-AI/lightning-Habana/pull/119))

### Changed

- Changed HPU docker image based on Synapse AI release 1.13.0 ([#114](https://github.com/Lightning-AI/lightning-Habana/pull/114))
-

### Fixed

- Fixed fabric imports for HPU strategies ([#126](https://github.com/Lightning-AI/lightning-Habana/pull/126))

### Removed

-

### Deprecated

-


## [1.2.0] - 2023-10-26

### Added

- Added tests, examples and documentation for HPUPrecisionPlugin with autocast ([#94](https://github.com/Lightning-AI/lightning-Habana/pull/94))
- Added test to validate checkpoint resuming with HPUDeepSpeedStrategy ([#95](https://github.com/Lightning-AI/lightning-Habana/pull/95))
- Added support for lightning 2.1 ([#100](https://github.com/Lightning-AI/lightning-Habana/pull/100), [#105](https://github.com/Lightning-AI/lightning-Habana/pull/105))

### Changed

- Changed HPU docker image based on synapse AI release 1.12.0 ([#90](https://github.com/Lightning-AI/lightning-Habana/pull/90))
- Use standard API's and Remove env variable to get HPU distributed backend ([#91](https://github.com/Lightning-AI/lightning-Habana/pull/91))
- Changed HPU docker image based on synapse AI release 1.12.1, updated hooks ([#106](https://github.com/Lightning-AI/lightning-Habana/pull/106))


### Fixed

-


### Removed

-


### Deprecated

-


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

### Fixed

- Fixed mnist example test ([#20](https://github.com/Lightning-AI/lightning-Habana/pull/20))
- Habana's dataloader hang with Lightning 2.0.x ([#29](https://github.com/Lightning-AI/lightning-Habana/pull/29))
- Make #29 applicable only for gaudi devices ([#39](https://github.com/Lightning-AI/lightning-Habana/pull/39))
- Fixed environment initialization for hpus and fixed docs ([#40](https://github.com/Lightning-AI/lightning-Habana/pull/40))
- Fixed docs and added work around to make use hpu media packages without signature issues ([#41](https://github.com/Lightning-AI/lightning-Habana/pull/41))

### Removed

- Cleaning up env's ID for HPU parallel plugins based on synapse AI release 1.9 ([#28](https://github.com/Lightning-AI/lightning-Habana/pull/28))
- Remove unnecessary import checks which degrade performance ([#38](https://github.com/Lightning-AI/lightning-Habana/pull/38))
