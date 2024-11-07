Here are the steps to execute the project titled "DOUBLE AUTHENTICATION SYSTEM   INTEGRATING FACE RECOGNITION AND EYE BLINK COUNT RECOGNITION".

step-1:

First need to install the python given in the Software Requirements file and the set path for python.

NOTE: While installing the python do not forgot to enable the "pip" command because we are using pip to install required packages to our project.

step-2:
 
Now install MySQL for database connectivity. Need to set username and password and the install the pycharm, it is IDE for python to run our project.

step 3:

After that open the project files folder with the help of pycharm and create virtual env for that project.

-To create virtual environment CLICK on the TERMINAL option present at the left corner of pycharm IDE and now a virtual env is created.

Step 4:

Now it's time to intall the required packages and type the following commands in line

-> pip install Flask

-> pip install opencv-python

-> pip install matplotlib

-> pip install numpy

-> pip install cvzone 

-> pip install pyyaml

-> pip install tabulate

After installing the above commands now all required python modules or packages will be available for our project.

Step 5:

At last create a database and table in that database with required attributes.

Follw the below steps to create a table...

-> login to MySQL as root:"mysql -u root"

-> Create a new database with a name of your wish:"GRANT ALL PRIVILEGES ON *.* TO 'db_user'@'localhost' IDENTIFIED BY 'provide your MySQL password here'"

-> log out of MySQL by typing: "\q"

-> log in as the new database user you just created: "mysql -u db_user -p"

-> now create a database: "CREATE DATABASE db_name"

-> now create a table in that database:"CREATE TABLE table_name"(place required attributes)
 
Hence the database and table creation is completed.

Step 6:

Now go to pycharm ide again and type "python test.py" and hit enter to run the project.

You will get link (local host) and click that link you will be redirected to LOGIN PAGE.

That's it, now you can use the project. 



