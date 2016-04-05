# Android Smartphone Programming - exercise grading manager
a tool to manage the exercise grading of the
*Android Smartphone Programming* seminar

## Features

* download exercise sheets from for different years from
    https://proglang.informatik.uni-freiburg.de/teaching/androidpracticum/
  and add them to database
* store exercises, students, gradings
* roughly validate the project structure of AS projects (android studio projects)
* automatically check for reports, ignored and required files
* change the name and label for launcher activities:<br>
  include student name, rep and exercise number
* parse students, grading schemes and more data from text-files
* comments per exercise, student, (sub)task with point modifiers
* inisible comments for Tutor/Lecturer
* output and commit text based gradings

## Requirements

### Linux

* Python 3
* Pip for python 3
* Python modules : peewee 2.6.4, wrapt, pyquery, dateutil  <!-- pyparsing -->

## Installation

No builds available, you gotta build it yourself for now..

### Linux (Ubuntu)

* Install python modules : 

        sudo apt-get install python3 python3-pip
        sudo pip3 install python-dateutil wrapt pyquery peewee==2.6.4

* check out this repository

        cd /your/install/path
        git clone https://github.com/IARI/asp_grader.git

<!--
* run
        cd /your/install/path
        make 
  in the directory where you checked out the repo
-->  

## Running

### Linux

run the script

	start.py

from the projects root directory.
	
## Feedback

If you have any questions, remarks or if you find a bug, please use the Issue Tracker.

## Usage

no instructions present yet.
