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

## Training: 