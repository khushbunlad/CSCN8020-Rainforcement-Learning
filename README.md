# CSCN8020-Rainforcement-Learning

### Project and environment setup

1. Move to project directory "CSCN8020" where you have cloned the project
2. Create virtual environment with name **"venvCSCN8020"**
    - Make sure ```python --version``` is set to **12.3.6** in your system
    - ```python -m venv venvCSCN8020```
3. Activate environment
    - ```.\venvCSCN8020\Scripts\Activate.ps1```
    - In case you are using visual studio code, Choose the environment from menu as active environment
4. Install packages mentioned in **"requirements.txt"**
    - ```pip install -r requirements.txt```
5. Select **"venvCSCN8020"** environment in your IDE
6. Create folder named **"Dataset"** in your project directory
7. Move all files downloaded from Kaggle dataset in the "Dataset" Directory


### Update Requirements.txt file once installing new packages

```pip freeze > requirements.txt```