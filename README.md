# TFG-Math-Chan-Vese




## Installation 🛠️

Follow these steps to set up the project:

### 1️⃣ Install Python 3.12

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
