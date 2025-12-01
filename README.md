# Implementation AI on Cheating Detection





## Table of Contents
- [Installation](#installation)
- [Usage](#usage)






## Installation: 

1. Make Sure you're using python version is 3.10.11
 If you don't have it, Go to https://www.python.org/downloads/release/python-31011/
 and download the Windows installer.

2. Create a Virtual Environment using python version 3.10.11 (recommended) \
   run this in terminal: \
   *For Windows users
   ```
    py -3.10 -m venv .venv
   .venv\Scripts\activate
   ```

    *For Mac users
   ```
   python3.10 -m venv .venv
   source .venv/bin/activate
   ```

3. After creating Virtual Environment, install all the requirement that is needed to run the system : \
   run this in terminal\
   *For Windows users
   ```
   pip install opencv-python==4.7.0.72 
   ```
   then install this,
   ```
   pip install --upgrade pip 
   pip install tensorflow==2.12 
   pip install opencv-python==4.7.0.72 
   pip install streamlit
   ```

   *For Mac users
   ```
   pip install --upgrade pip
   pip install opencv-python==4.7.0.72
   pip install tensorflow-macos==2.12
   pip install tensorflow-metal==1.0.0    # (optional)
   pip install streamlit
   ```

## Usage: 
Run it by \
```
streamlit run your_file_name.py
```
then head up to http://localhost:8501/

