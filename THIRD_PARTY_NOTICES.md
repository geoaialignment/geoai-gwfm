# Third-party notices

## Chronos-2 (Amazon)
GeoAI-GWFM-KR runs on the Chronos-2 time-series foundation model by Amazon Web Services.
The Chronos-2 architecture and pretrained weights are not included in this repository or in the
GeoAI-GWFM-KR adapter package. They are downloaded by the user from the official repository
https://huggingface.co/amazon/chronos-2 and are licensed under the Apache License, Version 2.0
(https://www.apache.org/licenses/LICENSE-2.0). The `chronos-forecasting` Python package
(https://github.com/amazon-science/chronos-forecasting) is also Apache-2.0. Nothing in the GeoAI-GWFM-KR
licenses restricts the rights that the Apache License grants for Chronos-2 itself.

Reference: Ansari, A. F., et al. (2025). Chronos-2: From univariate to universal forecasting. arXiv:2510.15821.

## Per-well recurrent baselines
`bench/baselines/rnn.py` is adapted from the code of Park, E., Park, J., & Kim, T. (2026), archived at
Zenodo, https://doi.org/10.5281/zenodo.23184760, under the PolyForm Noncommercial License 1.0.0.

## Data
Groundwater levels: National Groundwater Monitoring Network, National Groundwater Information Center,
Republic of Korea (https://www.gims.go.kr). Weather: Korea Meteorological Administration
(https://data.kma.go.kr). The data are not redistributed here; `bench/prepare.py` builds the benchmark
files from tables downloaded by the user.
