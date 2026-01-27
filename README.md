# Installation

## 1. Clone the Repository

Begin by cloning the repository to your local machine.

## 2. Install Package Dependencies

Navigate to the project directory and install the package dependencies listed in the `requirements.txt` file. This can typically be done using the following command:

```shell
pip install -r requirements.txt
```

## 3. Install libtorch Package

Since the Computer Platform is 11.7, you need to install the specific version of libtorch. Execute the following commands:

```shell
wget https://download.pytorch.org/libtorch/cu117/libtorch-shared-with-deps-2.0.1%2Bcu117.zip
unzip libtorch-shared-with-deps-2.0.1%2Bcu117.zip
```

Once the download and extraction are finished, you need to set $LIBTORCH_PATH environment variable with the actual path of your own libtorch installation:

```v
export LIBTORCH_PATH=/path/to/your/libtorch
```

This step is vital as it allows the project to access and utilize the libtorch library accurately.

## 4. Update Git Submodules

Navigate to the root folder of the project and perform the following operations:

```shell
git submodule update --init
```

## 5. Compiler and Install Project

Navigate to the root folder of the project and perform the following operations:

```shell
pip install -e .
```
