### Experiment workflow

To run this experiment on your local machine:

0. Download jsPsych from [here](https://github.com/jspsych/jsPsych/) and place the folder under `behavioral_experiment`.

1. navigate to the location where you want to create your project, and enter 
   ```
   git clone https://github.com/hawkrobe/tangrams.git
   ```
   at the command line to create a local copy of this repository. On Windows, run this command in the shell.

2. Make sure node.js and npm (the node package manager) are installed on your machine. Node.js sponsors an [official download](http://nodejs.org/download/) for all systems. For an advanced installation, there are good instructions [here](https://gist.github.com/isaacs/579814).

3. Run ```npm install``` at the command line in the `behavioral_experiment` directory to install dependencies. 

4. Finally, to run the experiment, run ```node app.js``` at the command line. You should expect to see the following message:
   ```
   info  - socket.io started
       :: Express :: Listening on port 8888
   ```
   This means that you've successfully created a 'server' that can be accessed by copying and pasting 
   ```
   http://localhost:8888/experiment.html
   ```
   in your browser. 
