# Dataset

The dataset will be distributed through a cloud-drive link. The download link and any extraction code will be added here once the upload is complete.

## Local layout

Extract the forward dataset to `data/rock_1000/`, so that sample directories are directly below it:

```text
data/rock_1000/
  sample_0000/
    data.pt
    meta.json
  sample_0001/
    data.pt
    meta.json
  ...
```

Each sample stores relative permittivity and B-scan observations; metadata specify acquisition geometry and numerical settings. The training and evaluation commands should use `--data-dir data/rock_1000`.

Dataset files and downloaded archives are excluded from Git. This directory tracks only the download and usage instructions. Archive size, checksum and exact contents will be documented with the download link.
