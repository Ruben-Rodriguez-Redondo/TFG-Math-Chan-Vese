# Chan-Vese Image Segmentation Implementation

This project presents a custom implementation of the Chan-Vese algorithm, originally introduced in the paper [*Active Contours Without Edges*](https://es.scribd.com/doc/36712164/10-1-1-2). 
It also includes extensions of the method to support segmentation with multiple evolving contours, based on the [*Chan-Vese Multi-Phase Segmentation Model*](https://www.ucd.ie/mathstat/t4media/EvanMurphy_SummerProjects2023_FinalReport.pdf) and to segment multichannel images, following the approach described in [*Active Contours Without Edges for Vector-Valued Images*](https://www.sciencedirect.com/science/article/abs/pii/S104732039990442X).

## Table of Contents

- [Installation](#installation-) 🛠️
- [Usage](#usage-) 💡
- [Interface](#-tkinter-interface) 📸
- [License](#license-) 📜


## Installation 🛠️

Follow these steps to set up the project:

### 1️⃣ Install Python 3.10

Make sure you have **Python 3.10** installed (probably other versions will work to). Can be downloaded from the
official site:  
🔗 [Python Downloads](https://www.python.org/downloads/)

To check your Python version, run in the local terminal:

```
python --version
```

### 2️⃣ Clone the repository

```
git clone https://github.com/Ruben-Rodriguez-Redondo/TFG-Math-Chan-Vese 
cd TFG-Math-Chan-Vese 
```

### 3️⃣ ️Install dependencies

Create a virtual environment, activated (consult the IDE or shell guide). In Pycharm is enough executing in local
terminal

```
python -m venv .venv
.venv\Scripts\activate
```

Once the environment is created and activated run:

`Proyect dependencies instalation`

```
python.exe -m pip install --upgrade pip
pip install . 
```

If the package is going to be modified run the following command to make the changes effective automatically.
```
pip install -e .
```

`Optional`: At least in my Pycharm community version (2024.2.5) even if the packages are correctly installed, and the
code works fine
some packages are still red-marked in the code as not installed. To avoid this warning (if happened to you) just add
.venv\Lib\Site-Packages to
your interpreter paths. More
details [here](https://stackoverflow.com/questions/31235376/pycharm-doesnt-recognize-installed-module).

## Usage 💡

The code is composed for two .py files the first one, chan-vese.py contains the logic and the implementation 
of the algorithm. A main example is provided, to see how it works run:

```
python chan_vese/chan-vese.py
```

The second one is an interface made with tkinter, which includes the interface along with its auxiliary functions (as allow the user to upload their own images). To access the interface run:

```
python chan_vese/interface.py 
```
It will be opened the window shown in the following section. In that window you can select the segmentation configuration in a simpler way that would be using  chan_vese.py directly. 


## 📸 Tkinter interface

Appears after running interface.py, here you can select the Chan-Vese segmentation algorithm params, hyperparams and the image to segment. 
<div align="center">
  <img  src = "/figs/interface.png" alt = "Main Functionalities">
</div>
This one appears once  the green button (Chan-Vese segmentation) is pressed. It keeps  a real time updating of the segmentation process, also informing when is over and how long it took.
<div align="center">
  <img  src = "/figs/interface_2.png" alt = "Main Functionalities">
</div>

## License 📜

This work is licensed under
the [Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).
You are free to copy, modify, distribute, and reproduce the material in any medium, provided that you give appropriate
credit, indicate if changes were made, and distribute your contributions under the same license.

<div align="center">
  <img src="/figs/license.png" alt="License">
</div>
