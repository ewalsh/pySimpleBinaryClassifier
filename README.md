# pySimpleBinaryClassifier
A Simple Binary Classifier in Python

# Some notes on style
This project uses python 3.10 and case match statements. It also attempts to use functional style and avoid pandas in favour of more basic data structures where possible in this quick edition.

These ideas are stylistic but good programming practice from my perspective. For example, in this project, not much is known about the variables or how they will eventually be ingested. In my opinion, using a functional style is cleaner and the simpler data structures will make conversion to streaming or data base connections easier in future. It also allows for more type saftey using the typing within python.

# Structure of app
While the focus of this app is on binary classification, the structure and design are much more focused on setting up an app that we can test, test, and re-test.

There is a lot that we don't know within this project, for example, you can see that even the variables are generically named. This matters because it doesn't really provide the opportunity to use theory when feature engineering. Also, there are several variables that could be categorical or just very discrete numerical values.

For these reasons, the structure of this app is designed so that different modelling and feature engineering decisions can be altered and re-tested. The purpose of this is to make a general flow/process designed app that can be continue to be extended. As such, less time was spent on tuning any particular model but rather a setup that allows for flexible approaches and models to
be tried and compared.

# Using the app
First, please make sure you have python 3.10 available. You can see a copy of the .env file within exampleEnv.txt. Please copy that into the root directory as .env. This is where model and feature engineering changes can be made. Also, it is best to use a virtual environment and the requirements.txt file.

After this setup is ready, you can run the app with a simple:
`python app.py`
comand from the root directory.
