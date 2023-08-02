# BarbellPhysics
The aim of this project is to launch a server (for now on localhost) where people can upload media (server.js + index.html), 
have it analyzed through a Python child process to detect Mediapipe landmarks and squat form accuracy (analyze_squat.py), and then displayed for the user to see (work in progress). 
During processing, video, landmark, form, and rep data is stored in a MongoDB database. Uploads and Processed_Videos are stored in their respective folders

Notes for usage: 
Make sure to change the IP address to your IP for server.js
To access MongoDB, change <Password> for the actual password for the uri (analyze_squat.py)
Some errors that may occur for analyze_squat will not show on the terminal, such as in the try-execpt or if the file directory is not found, render_video() will just skip over it. If landmarks are not being shown where
there clearly is a person, then there is probably an error inside of the try except. 
All necessary Python and Node.js modules are installed, but you may need to install them globally to run the python child process from server.js
Hello.py shows an example of how to pass parameters from node.js to the python child process. To test, uncomment the section in server.js and change the line to:
  <const python = spawn('hello.py', ['analyze_squat.py', 'param1', 'param2'])>


TODO in order of priority: 
1. Fix child process so that the user can view the end product. See how OpenCV (video runner and handler) responds to being called as a child process
2. Look for alternatives to OpenCV or find a way to make it run through each frame faster
3. Try out models using world_landmarks and using the landmarks for just the body (11-33). Fix RidgeClassifier Model display. Send each model's prediction (good or bad?)
4. Work on training machine learning model for deadlifts
