# HI Exchange Interface

External HI files can be placed here to override the default V1b HI.

## Usage

Place CSV files in this directory with the following naming convention:

```
hi_input/
  Bearing1.csv       ← Train Bearing 1 HI
  Bearing2.csv       ← Train Bearing 2 HI
  Bearing3.csv       ← Train Bearing 3 HI
  Bearing4.csv       ← Train Bearing 4 HI
  Test1.csv          ← Test 1 HI
  Test2.csv          ← Test 2 HI
  Test3.csv          ← Test 3 HI
  Test4.csv          ← Test 4 HI
  Test5.csv          ← Test 5 HI
  Test6.csv          ← Test 6 HI
```

## CSV Format

Each file must have an `HI` column:

```csv
HI
0.000
0.012
0.025
...
```

Additional columns (e.g., `file_idx`, `cycle`) are allowed and ignored.

## How it works

When running any experiment with `--use-hi-input` flag (or setting `HI_INPUT_DIR`),
the loader checks for these files first. If found, they override the default V1b HI.
If not found, the default V1b HI is used as fallback.

Example:
```bash
python run_expB.py --hi-input /path/to/hi_input/
```
