# MICCAI 2022 - Paper ID: 2116

## Setting up the OS:

- This code was develop and runs properly in Ubuntu 18.04 and with Python 3.6.9

```
sudo apt update
```

- In case `pip3` is not install, then run:

```shell
sudo apt install python3-pip
```

- Install `pipenv`:

```shell
pip3 install --user pipenv
```

## Setting up the environment and dependencies:

- Inside the main directory (`miccai-2022`), run the following command to syncronize all the dependencies:

```shell
pipenv sync
```

- To activate the virtual environment:

```shell
pipenv shell
```

## Running the experiments: 
### **Experiment 01:**
This experiment uses the 8 WSI images in the dataset and the Macenko color normalization method.

- For the 128x128 patch-size dataset, with cross validation and cross testing:
```shell
sh test_00_128x128.sh
sh test_01_128x128.sh
sh test_02_128x128.sh
sh test_03_128x128.sh
```

- For the 256x256 patch-size dataset, with cross validation and cross testing:
```shell
sh test_00_256x256.sh
sh test_01_256x256.sh
sh test_02_256x256.sh
sh test_03_256x256.sh
```

### **Experiment 02:**
This experiment evaluates the impact of the scanner in the segmentation of plaques. We use only the 128x128 patch-size dataset as we obtained better performance with this patches in the first experiment.

- To train using the WSI from the scanner Hamamatsu NanoZoomer 2.0-RS:
```shell
sh test_00_128x128_oldscan.sh
```

- To train using the WSI from the scanner Hamamatsu NanoZoomer S60 sc:
```shell
sh test_00_128x128_newscan.sh
```

### **Experiment 03:**
This experiment evaluates the impact of the color normalization. We use only the 128x128 patch-size dataset and the configuration of the best fold from the first experiment.

```shell
sh test_00_128x128_bestfold_vahadane.sh
```

### Full results:
### **Experiment 01:**

- The following table shows the results for every fold in experiment 01. Best fold is reported in bold font.

|   fold_name   |   |  dev_dice  | dev_f1 | dev_recall | dev_precision |   | test_dice | test_f1 | test_recall | test_precision |
|:-------------:|:-:|:----------:|:------:|:----------:|:-------------:|:-:|:---------:|:-------:|:-----------:|:--------------:|
|               |   |            |        |            |               |   |           |         |             |                |
| test_00_cv_00 |   |   0.7151   | 0.7165 |   0.6674   |     0.8017    |   |   0.6753  |  0.6707 |    0.668    |      0.784     |
| test_00_cv_01 |   |   0.7034   | 0.6933 |    0.699   |     0.718     |   |   0.7046  |  0.7035 |    0.7495   |     0.7475     |
| test_00_cv_02 |   |   0.6963   | 0.6932 |   0.6873   |     0.7339    |   |   0.6781  |  0.6765 |    0.7423   |     0.7094     |
| test_01_cv_00 |   |   0.7011   | 0.7037 |    0.714   |     0.7239    |   |   0.7032  |  0.6962 |    0.6684   |     0.8052     |
| test_01_cv_01 |   |   0.7231   | 0.7118 |   0.6763   |     0.7801    |   |   0.7248  |  0.7192 |    0.7105   |     0.8067     |
| test_01_cv_02 |   |    0.72    | 0.7217 |   0.7519   |     0.7185    |   |   0.7141  |  0.7068 |    0.6811   |     0.8166     |
| test_02_cv_00 |   |    0.709   | 0.7156 |   0.7195   |     0.7423    |   |   0.7027  |  0.7004 |    0.7043   |     0.7855     |
| test_02_cv_01 |   |   0.7127   | 0.7195 |   0.7012   |     0.7618    |   |   0.6643  |  0.6608 |    0.6444   |     0.7996     |
| test_02_cv_02 |   |   0.6807   | 0.6838 |   0.6862   |     0.7167    |   |   0.6306  |  0.6296 |    0.6634   |     0.7316     |
| test_03_cv_00 |   |   0.6765   | 0.6813 |   0.6788   |     0.7183    |   |   0.6845  |  0.6855 |    0.8061   |      0.666     |
| test_03_cv_01 |   |   0.6959   | 0.6967 |   0.6244   |     0.8167    |   |   0.6883  |  0.879  |    0.7981   |     0.6777     |
| test_03_cv_02 |   |   0.6112   | 0.6008 |    0.61    |     0.6325    |   |   0.6521  |  0.6545 |    0.8018   |     0.6245     |
|               |   |            |        |            |               |   |           |         |             |                |
|      mean     |   |   0.6954   | 0.6948 |   0.6847   |     0.7387    |   |   0.6852  |  0.6986 |    0.7198   |     0.7462     |
|      std      |   |   0.0289   | 0.0313 |   0.0373   |     0.0462    |   |   0.0260  |  0.0596 |    0.0561   |     0.0615     |
|      max      |   |   0.7231   | 0.7217 |   0.7519   |     0.8167    |   |   0.7248  |  0.879  |    0.8061   |     0.8166     |
|      min      |   |   0.6112   | 0.6008 |    0.61    |     0.6325    |   |   0.6306  |  0.6296 |    0.6444   |     0.6245     |

### **Experiment 02:**

