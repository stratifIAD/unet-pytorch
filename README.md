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
This uses the 8 WSI images in the dataset and the Macenko color normalization method.

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