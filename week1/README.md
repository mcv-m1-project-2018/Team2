# Traffic Sign Detection

## Usage

```
usage: traffic_sign_detection.py [-h] [--test TEST] [--output OUTPUT]
                                 [--window-method WINDOW_METHOD] [--analysis]
                                 [--threads THREADS] [--executions EXECUTIONS]
                                 train_path pixel_methods

Detect traffic signs in images using non ML methods.

positional arguments:
  train_path            Training directory
  pixel_methods         Methods to use separated by ";". If using --output,
                        pass a single param. Valid values are method1,
                        method2, method3 and method4

optional arguments:
  -h, --help            show this help message and exit
  --test TEST           Test directory, if using test dataset to generate
                        masks.
  --output OUTPUT       Output directory, if using test dataset to generate
                        masks.
  --window-method WINDOW_METHOD
                        Window method to use.
  --analysis            Whether to perform an analysis of the train split
                        before evaluation. Train mode only.
  --threads THREADS     Number of threads to use. Train mode only.
  --executions EXECUTIONS
                        Number of executions of each method. Train mode only.

```

## Methods

### Method 1

- Color segregation using RGB colorspace.
- Blur
- Fill holes.
- Remove noise using geometry.

### Method 2

- Color segregation using HSV colorspace.
- Blur
- Fill holes.
- Remove noise using geometry.

### Method 3

- White balance using histogram equalization.
- Color segregation using HSV colorspace.
- Blur
- Fill holes.
- Remove noise using geometry.

### Method 4

- White balance using adaptive histogram equalization.
- Color segregation using HSV colorspace.
- Blur
- Fill holes.
- Remove noise using geometry.
