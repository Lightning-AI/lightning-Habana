# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - YYYY-MM-DD

### Added

- Added HPU support for fabric ([#11](https://github.com/Lightning-AI/lightning-Habana/pull/11))
- Added Pytorch HPU profiler support ([#15](https://github.com/Lightning-AI/lightning-Habana/pull/15))

- Added basic HPU infra support for deep speed ([#21](https://github.com/Lightning-AI/lightning-Habana/pull/21))

- Added Pytorch HPU datamodule support ([#16](https://github.com/Lightning-AI/lightning-Habana/pull/16))

### Changed

- Changed code hierarchy in compliance with base lightning code for pytorch ([#12](https://github.com/Lightning-AI/lightning-Habana/pull/12))
-
### Fixed

- Fixed mnist example test ([#20](https://github.com/Lightning-AI/lightning-Habana/pull/20))
- Habana dataloader hang with Lightning 2.0.x ([#29](https://github.com/Lightning-AI/lightning-Habana/pull/29))

### Removed

- Cleaning up env's ID for HPU parallel plugins based on synapse AI release 1.9 ([#28](https://github.com/Lightning-AI/lightning-Habana/pull/28))

### Deprecated
