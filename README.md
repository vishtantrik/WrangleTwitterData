# WrangleTwitterData

## Introduction
In this project I wrangle (gather, assess, and clean) and analyze
tweets by WeRateDogs (@dog_rates), a Twitter account that rates 
people's dogs with a humorous comment about the dog.
 
I worked on this project as part of the Udacity Data Analyst 
Nanodegree. Apart from learning and implementing data wranging in 
Python, I used this excercise to explore the dataset and check my
intution. Among other things I found that the “French Bulldog” is the
most popular dog breed.

## Getting Started
The code is in Python 3. All the code is in the _wrangle_act.py_ file.

### Data Files
One of the pieces of data needed for this project,
_twitter-archive-enhanced.csv_, was supplied by Udacity to me. While I
await confirmation from them to include this file in the repo I have 
replaced it with a sample file of the same name. If/when I get a 
response from them I will post the full file. **Till then this project 
will not be fully replicable using this repo and will throw exceptions
:(** .

### Twitter API
You will need to replace the placeholders for the twitter API 
credentials (consumer_key, consumer_secret, access_token, 
access_token_secret) with your own credentials.


## Code Structure
The code is structured as follows:
* Data Wrangling
    * Gathering
    * Assessing
        * Code
        * List Quality Issues
        * List Tidiness Issues
    * Cleaning
        * Missing Data
        * Tidiness 
        * Other Quality Issues
* Data Storage
* Analysis and Visualization

## Documentation
2 reports are included in the repository:
* _act_report.pdf_ : Introduces the project and documents the aspects I
 explored in the data and some of the findings
* _wrangle_report.pdf_ : Documents the steps followed in "wrangling" 
the data including some of the challenges I faced

## Authors
* [**Vishruth Srinath**](https://www.linkedin.com/in/vishruthsrinath/)

## Acknowledgements
* Udacity Data Analyst Nanodegree program for:
     * Doing an excellent job of teaching Python for data wrangling
     * Laying out a basic outline of this project
     * Supplying the enhanced tweet data and the image prediction data

## Notes
* This project was originally done in a Jupyter notebook. There are 
still some hold-overs such as in-line plots, and outputs from commands
such as the pandas "head()" command in the code. I plan to clean this 
up or find a way to have the notebook itself in the repo soon.
