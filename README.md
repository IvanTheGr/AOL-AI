# Implementation AI on Cheating Detection
This project aims to develop an AI-powered cheating detection system designed for online examinations. 
The system uses a combination of TensorFlow, OpenCV, and Streamlit to monitor user behavior in real time. 
Without relying on Mediapipe or any complex external dependencies, the program focuses on lightweight, fast, and flexible computer-vision-based monitoring.




## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)


## Installation 

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

## Usage 
Run it by 
```
streamlit run your_file_name.py
```
- then head up to http://localhost:8501/ \
- Next it will show login page \
- Use the following credentials to log in: \
Username: admin \
Password: 1234 \
- Click the "Start Webcam" button to begin monitoring. \
- The application will display the webcam feed and log any suspicious behavior. \
- The application also detect user facial expression and will write it on log. \
- The application will collect 3 warnings that is caused by user during the exams, could be by \
- head and eye gaze (up, left, right and bottom) for 10 seconds and away from the screen for 10 seconds either. \
- If user is doing head/eye gaze or maybe away from the screen, the application will detect user is on suspicious behavior. \
- The application will gave a warning for user, based on it's causal activities. \
- If 3 warnings is reached, the application will auto lock, it's mean that user can't access the application anymore until the exam done. \
- User's suspicious behavior will be saved as image and recording, which will be the evidence during the examination. \
- Then by clicking stop webcam, all the evidence will be saved locally on the host file explorer in folder form. \


## Contributing 
Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request. 


## License 
This project is licensed under the MIT License.  \
You are free to use, modify, and distribute this project for both personal and commercial purposes, as long as proper credit is given.

See the LICENSE file for more details.
