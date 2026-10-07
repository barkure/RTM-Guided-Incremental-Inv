# Dataset

Download the dataset archive from [iCloud Drive](https://www.icloud.com.cn/iclouddrive/0a37XwGrQKcWwBEGBawKtLvLg).

- Archive: `rock_1000.zip`
- Size: approximately 776 MiB
- SHA-256: `5e894bbb248589bf1eeab04bd633fd222d53263df758fd10dfa13131e3583465`

Verify the downloaded archive before extraction:

```bash
sha256sum rock_1000.zip
# macOS: shasum -a 256 rock_1000.zip
```

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

Dataset files and downloaded archives are excluded from Git. This directory tracks only the download and usage instructions. The archive contains the top-level `rock_1000/` directory, sample directories and a dataset manifest.
