
# coding: utf-8

# # Wrangling and Analyzing WeRateDogs' Tweets

# In this notebook I will be wrangling and analyzing tweets by WeRateDogs (@dog_rates).
# WeRateDogs is a Twitter account that rates people's dogs with a humorous comment about the dog.
# 
# The notebook will be structured as follows:
# 
# * Data Wrangling
#     * Gathering
#     * Assessing
#         * Code
#         * List Quality Issues
#         * List Tidiness Issues
#     * Cleaning
#         * Missing Data
#         * Tidiness 
#         * Other Quality Issues
# * Data Storage
# * Analysis and Visualization

# In[468]:


import pandas as pd
import numpy as np
import requests
import tweepy
import json
import time
import datetime

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt


# In[469]:


consumer_key = "<consumer_key>"
consumer_secret = "<consumer_secret>"
access_token = "<access_token>"
access_token_secret = "<access_token_secret>"


download_tweet_json = False


# ## Data Wrangling
# 
# Data wrangling consists of gathering, assessing, and cleaning data to enable analysis and visualizations.
# This section will go through the above three steps.

# ### Gathering
# The data is to be gathered from three (3) sources:
# 
# 1. Basic and derived tweet data from the WeRateDogs Twitter archive provided to us as a CSV file on the hard disk: `twitter_archive_enhanced.csv` 
# 2. The tweet image predictions in a TSV file [hosted by Udacity](https://d17h27t6h515a5.cloudfront.net/topher/2017/August/599fd2ad_image-predictions/image-predictions.tsv): `image_predictions.tsv`
# 3. Detailed tweet data from the `twitter API` and the tweepy library

# #### Archive Data
# 
# This data will be gathered from a CSV file in the same location as this notebook

# In[470]:


archive = pd.read_csv("twitter-archive-enhanced.csv")
archive.head(3)


# #### Image Predictions
# Data is stored in a TSV file hosted by Udacity

# In[471]:


file_url = r"https://d17h27t6h515a5.cloudfront.net/topher/2017/August/599fd2ad_image-predictions/image-predictions.tsv"


# In[472]:


# write url to file
file_name = file_url.split("/")[-1]
content = requests.get(file_url).content
with open(file_name, mode ="wb") as file:
    file.write(content)


# In[473]:


# read from file to dataframe
image_predictions = pd.read_csv(file_name, sep = "\t")
image_predictions.head()


# #### Tweet Details

# In[474]:


# Setup API connection
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

twitter_api = tweepy.API(auth, wait_on_rate_limit = True,
                        wait_on_rate_limit_notify = True)


# In[475]:


# Get tweet for each tweet ID  and write to file
tweet_ids = list(archive.tweet_id)
file_name = "tweet_json.txt"


# In[476]:


num_ids = len(tweet_ids)
count = 0
update_res = 20
if(download_tweet_json):
    start_time = time.time()
    missing_ids = []
    with open(file_name, mode = "w") as file:
        for tweet_id in tweet_ids:
            count += 1
            # Get tweet
            try:
                tweet = twitter_api.get_status(tweet_id, tweet_mode = "extended")
                # Write to file
                json.dump(tweet._json, file)
                file.write("\n")
            except tweepy.TweepError as te:
                print("Missing tweet id: {0}".format(tweet_id))
                missing_ids.append(tweet_id)
            # Show progress
            if count % update_res == 0:
                current_time = time.time()
                print("{0}/{1} Complete, running for {2:.1f} s".format(
                    count, num_ids, current_time - start_time))

    end_time = time.time()
    print("Complete, total time to get Tweet details : {0:.1f} s".format(
                    count, num_ids, end_time - start_time))


# In[477]:


# Parse JSON from file, line by line
df_list = []
with open(file_name, mode = "r", encoding = "UTF-8") as json_file:  
    for line in json_file:
        data = json.loads(line)
        df_list.append({"tweet_id": data["id"],
                       "retweet_count": data["retweet_count"],
                       "favorite_count": data["favorite_count"],
                        "retweeted": data["retweeted"],
                       "favorited": data["favorited"],
                       "created_at": data["created_at"]})        


# In[478]:


# Create DataFrame
tweet_details = pd.DataFrame(df_list, columns = ["tweet_id", "retweet_count", "favorite_count",
                                                "retweeted", "favorited", "created_at"])
tweet_details.head()


# ### Assessing

# #### Code
# ##### `archive` Table

# In[479]:


archive.info()


# In[480]:


archive[archive.in_reply_to_status_id.notnull()].head()


# In[481]:


int(archive.in_reply_to_status_id.iloc[30])


# In[482]:


archive.puppo.value_counts()


# In[483]:


archive.doggo.value_counts()


# In[484]:


archive.floofer.value_counts()


# In[485]:


archive.pupper.value_counts()


# In[486]:


sum((archive.pupper == "pupper") & (archive.doggo == "doggo"))


# In[487]:


archive[(archive.pupper == "pupper") & (archive.doggo == "doggo")].head()


# In[488]:


sum((archive.pupper == "pupper") & (archive.floofer == "floofer"))


# In[489]:


archive.head()


# In[490]:


archive.rating_denominator.value_counts()


# In[491]:


for i, row in archive[archive.rating_denominator != 10].iterrows():
    print(row.text)
    print("{0} / {1}".format(row.rating_numerator, row.rating_denominator))


# In[492]:


archive.rating_numerator.value_counts()


# In[493]:


def printOriginal_RatingFilter(numerator = 10000):
    '''
    Print the text, numerator and denominator for all
    original tweets that have a numerator < that specified
    '''
    mask = archive.in_reply_to_status_id.isnull()
    mask = mask & (archive.retweeted_status_id.isnull())
    mask = mask & (archive.rating_numerator == numerator)
    for i, row in archive[mask].iterrows():
        print(row.text)
        print("{0} / {1}".format(row.rating_numerator, row.rating_denominator))


# In[494]:


printOriginal_RatingFilter(0)


# In[495]:


printOriginal_RatingFilter(1)


# In[496]:


printOriginal_RatingFilter(2)


# In[497]:


printOriginal_RatingFilter(3)


# In[498]:


sum(archive.duplicated())


# In[499]:


archive.sample(10)


# In[500]:


archive.name.head()


# In[501]:


archive.name.value_counts()


# In[502]:


for index, row in archive.sample(10).iterrows():
    print(row.text)
    print(row["name"])


# ##### `image_predictions` Table

# In[503]:


image_predictions.info()


# In[504]:


image_predictions.head()


# In[505]:


image_predictions.sample(10)


# In[506]:


image_predictions[image_predictions.p1_dog].p1.value_counts()


# In[507]:


image_predictions[image_predictions.p2_dog].p2.value_counts()


# #### `tweetdetails` Table

# In[508]:


tweet_details.info()


# In[509]:


tweet_details.head()


# In[510]:


tweet_details.sample(10)


# #### List Quality Issues
# ##### `archive` table
# 
# * Bad datatypes (tweet_id, timestamp, in_reply_to_status_id , in_reply_to_user_id, retweeted_status_id, retweeted_status_user_id,
# doggo, pupper, floofer, puppo)
# * Incorrect ratings : 9/11, 11/15, 960/0, 4/20 x 2, 50/50, 1/2
# * Many ratings are not for dogs
# * Includes retweets and replies to tweets
# * Many incorrect occurences of "a", "an", and "the" as dog names
# * When no name is found the string "None" is used in the "name" column
# * Does not record dog names from sentences structured like:  "... named -dog name-" and "...... this is -dog name-"
# 
# ##### `image_predictions` Table
# 
# * Breed names sometimes lowercase, sometimes upper case.
# * Breed names have "_" instead of " "
# * Bad datatype (tweet_id)
# 
# ##### `tweet_details` Table
# 
# * Bad datatype (created_at, retweet_count, favorite_count)
# 

# ##### List Tidiness Issues
# 
# * All tables, "archive", "image_predictions", and "tweet_details", contain same type of data
# * Duplicated data, "timestamp" column in "archive" table and "created_at" columns in "tweet_details" table
# * "Doggo", "Pupper", "Puppo", "Floofer" are the same variable, i.e. dog stage, but are in 4 columns

# ### Cleaning

# In[511]:


archive_clean = archive.copy()
image_predictions_clean = image_predictions.copy()
tweet_details_clean = tweet_details.copy()


# #### Missing Data

# ##### `archive` Table: Does not record dog names from sentences structured like:  "... named -dog name-" and "...... this is -dog name-"
# 
# ###### Define
# Use str.extract and regex on tweet text to extract names from sentences whose structure is as above and update dog name, if there is no name currently
# 
# ###### Code

# In[512]:


tweet_mask = archive_clean.name == "None"

# Extract
pattern = r"\w* (named|this is) (?P<dog_name>\w*)"
new_names = archive_clean.text.str.extract(pattern, expand = True)
# Print for inspection
new_names[new_names.dog_name.notnull()]


# Looks like "... this is -dog name-" produces a few new dog names but many more other words. Hence I will not look for dog names in such sentences.

# In[513]:


archive.loc[1853].text


# In[514]:


# Extract
pattern = r"\w* named (?P<dog_name>\w*)"
new_names = archive_clean.text.str.extract(pattern, expand = True)
# Print for inspection
new_names[new_names.dog_name.notnull()]


# In[515]:


# Covert "None", "a", "an", "the" in dog name to "NaN" to make merging easier
mask = (archive_clean.name == "None")
mask = mask | (archive_clean.name == "a")
mask = mask | (archive_clean.name == "an")
mask = mask | (archive_clean.name == "the")


# In[516]:


archive_clean.loc[mask, "name"] = np.nan
# Test
archive_clean.name.value_counts().head()


# In[517]:


# Merge archive and new names
archive_clean = archive_clean.merge(new_names, how = "left", left_index = True, right_index = True)
archive_clean.info()


# In[518]:


# Replace
mask = archive_clean.name.isnull()
archive_clean.loc[mask, "name"] = archive_clean.loc[mask, "dog_name"]
# Drop "dog_name" column
archive_clean.drop(labels = "dog_name", axis = "columns", inplace = True)
archive_clean.info()


# ###### Test

# In[519]:


mask = (archive.name == "None")
mask = mask | (archive.name == "a")
mask = mask | (archive.name == "an")
mask = mask | (archive.name == "the")
archive.loc[~mask,["name"]].info()


# In[520]:


archive_clean.loc[:, ["name"]].info()


# #### Tidiness

# ##### All tables, `archive`, `image_predictions` and `tweet_details`, contain same type of data
# ###### Define
# Merge all tables on the "tweet_id"

# ###### Code

# In[521]:


archive_master_clean = pd.merge(left = archive_clean, right = tweet_details,
                                how = "outer", on = "tweet_id")
# data notes say that we only need tweets which have images, so a "right" merge is performed below
archive_master_clean = archive_master_clean.merge(right = image_predictions, how = "right", on = "tweet_id")


# ###### Test

# In[522]:


archive_clean.info()


# In[523]:


archive_master_clean.info()


# In[524]:


archive_master_clean.shape


# Columns from all three tables should be present. There should be only 1 "tweet_id" column.
# 
# Also, number of rows should be same as number of "jpg_url", i.e. 2075

# ##### Duplicated data, "timestamp" column in "archive" table and "created_at" columns in "tweet_details" table
# ###### Define
# Drop "created_at" column in the `archive_master_clean` table

# ###### Code

# In[525]:


archive_master_clean.drop(labels = "created_at", axis = "columns", inplace = True)


# ###### Test

# In[526]:


archive_master_clean.info()


# The "created_at" column should not be shown above

# ##### "Doggo", "Pupper", "Puppo", "Floofer" are the same variable, i.e. dog stage, but are in 4 columns
# ###### Define
# * For each row, Append non-"None" values in all 4 columns together to create a column "dog_stage". This makes sure we do not miss out on any tweets with multiple stages
# * Replace empty ( == "") values with np.nan
# * Drop the original 4 columns

# ###### Code

# In[527]:


# Append to create "dog_stage"
def CreateDogStage(row):
    dog_stage = ""
    # Doggo
    if(row.doggo != "None"):
        dog_stage = "doggo"
    
    # Puppo
    if(row.puppo != "None"):
        if(dog_stage == ""):
            dog_stage = "puppo"
        else :
            dog_stage = dog_stage + "&" + "puppo"
    
    # Pupper
    if(row.pupper != "None"):
        if(dog_stage == ""):
            dog_stage = "pupper"
        else :
            dog_stage = dog_stage + "&" + "pupper"
            
    # Floofer
    if(row.floofer != "None"):
        if(dog_stage == ""):
            dog_stage = "floofer"
        else :
            dog_stage = dog_stage + "&" + "floofer"
    
    
    return(dog_stage)


# In[528]:


dog_stage_list = ["doggo", "pupper", "puppo", "floofer"]
archive_master_clean["dog_stage"] = archive_master_clean[dog_stage_list].apply(
    CreateDogStage, axis = 1)
archive_master_clean.loc[(archive_master_clean["dog_stage"] == ""), "dog_stage"] = np.nan


# In[529]:


# Drop orignial columns
archive_master_clean.drop(labels = dog_stage_list, axis = "columns", inplace = True)


# ###### Test

# In[530]:


archive.merge(right = image_predictions, how = "right").replace(
    {"doggo": {"None" : np.nan}, "puppo": {"None" : np.nan},
     "pupper": {"None" : np.nan}, "floofer": {"None" : np.nan}}).dropna(
    subset = dog_stage_list, how = "all").shape


# In[531]:


sum(archive_master_clean.dog_stage.notnull())


# The number of rows in the archive.merge....replace......info() shuld be the same as the number of non-null "dog_stage"s

# In[532]:


archive_master_clean.info()


# The columns "doggo", "pupper" and "puppo" shouldn't exist

# In[533]:


archive_master_clean.dog_stage.value_counts()


# #### Other Quality Issues
# 
# ##### `archive` table : Includes retweets and replies to tweets
# ###### Define
# * Use non-null "retweeted_status_id" values as proxy for a tweet being a retweet and non-null "in_reply_to_status_id" values as a proxy for a tweet being a reply to a tweet to drop rows
# * Drop columns "in_reply_to_status_id", "in_reply_to_user_id", "retweeted_status_id", "retweeted_status_user_id", "retweeted_status_timestamp"

# ###### Code

# In[534]:


# Drop rows
mask = archive_master_clean.retweeted_status_id.isnull()
mask = mask & (archive_master_clean.in_reply_to_status_id.isnull())
archive_master_clean = archive_master_clean[mask]


# ###### Test

# In[535]:


archive_master_clean.info()


# There should be 0 non-null entries in the "in_reply..." and "retweeted_..." columns

# ###### Code

# In[536]:


# Drop columns
archive_master_clean = archive_master_clean.drop(labels = ["in_reply_to_status_id", "in_reply_to_user_id",
                                    "retweeted_status_id", "retweeted_status_user_id", "retweeted_status_timestamp"],
                         axis = "columns")
archive_master_clean.reset_index(inplace = True, drop = True)


# ###### Test

# In[537]:


archive_master_clean.info()


# The dropped columns should not exist

# ##### `archive` table : Many incorrect occurences of "a", "an", and "the" as dog names
# ###### Define
# This has already been addressed
# 
# ###### Test

# In[538]:


sum(archive_master_clean.name == "a")


# In[539]:


sum(archive_master_clean.name == "an")


# In[540]:


sum(archive_master_clean.name == "the")


# Te above sums should all be zero

# ##### `archive` table : When no name is found the string "None" is used in the "name" column
# ###### Define
# Has already been addressed

# ###### Test

# In[541]:


sum(archive_master_clean.name == "None")


# The above sum should be "0"

# ##### `archive` table : Bad datatypes (timestamp, in_reply_to_status_id , in_reply_to_user_id, retweeted_status_id, retweeted_status_user_id, doggo, pupper, floofer, puppo)
# ###### Define
# 
# * Convert tweet_id to string
# * Convert timestamp to pd.datetime using pd.to_datetime
# 
# Other columns don't exist anymore

# ###### Code

# tweet_id

# In[542]:


archive_master_clean.tweet_id = archive_master_clean.tweet_id.astype(str)


# timestamp

# In[543]:


archive_master_clean.timestamp.head(2)


# In[544]:


# check if all time offsets are +0000
archive_master_clean.timestamp.str.slice(start = 20, stop = 25).value_counts()


# In[545]:


# timestamp column
dt_format = "%Y-%m-%d %H:%M:%S"
utc_flag = True # based on above value counts and twitter data-dictionary
archive_master_clean.loc[:, "timestamp"] = pd.to_datetime(archive_master_clean.timestamp,
                                                          format = dt_format, utc = utc_flag)


# ###### Test

# In[546]:


archive_master_clean.info()


# datatype for timestamp should be "datetime" and "tweet_id" should be "object"

# In[547]:


archive_master_clean.timestamp[0]


# In[548]:


archive[archive.tweet_id == archive_master_clean.tweet_id.astype("int")[0]].timestamp


# Time stamps should match

# `tweet_details` table: Bad datatypes (created_at, retweet_count, favorite_count should be integer not floats)
# ###### Define
# 
# * Convert retweet_count, favorite_count to int from float
# 
# Other columns don't exist anymore
# 
# ###### Code

# In[549]:


# Fill NaN's with 0 as integer type cannot have NaN values
archive_master_clean.favorite_count.fillna(0, inplace= True)
archive_master_clean.retweet_count.fillna(0, inplace= True)
# Cast as integer
archive_master_clean.favorite_count = archive_master_clean.favorite_count.astype("int")
archive_master_clean.retweet_count = archive_master_clean.retweet_count.astype("int")


# ##### Test

# In[550]:


archive_master_clean.info()


# retweet_count and favorite_count should be of type "int64"

# ##### `archive` table : Incorrect ratings : 9/11, 11/15, 960/0, 4/20 x 2, 50/50, 1/2
# ###### Define
# Inspect these and change the rating numerator and denominator on a case-by-case basis
# 
# ###### Code

# In[551]:


def GetTweetId(df, numerator, denominator):
    mask = df.rating_numerator == numerator
    mask = mask & (df.rating_denominator == denominator)
    tweet_id = list(df.loc[mask, "tweet_id"])
    return(tweet_id)


# In[552]:


def PrintTextRatings(df, this_ids):
    for this_id in this_ids:
        mask = (df.tweet_id == this_id)
        print(df.text[mask].iloc[0])
        print("Stored Rating = {}/{}".format(df.loc[mask, "rating_numerator"].iloc[0], df.loc[mask, "rating_denominator"].iloc[0]))


# In[553]:


def ResetRatings(df, this_id, numerator, denominator):
    if len(this_id) > 1:
        raise ValueError("There are more than one tweet ids")
    mask = (df.tweet_id == this_id[0])
    df.loc[mask, "rating_numerator"] = numerator
    df.loc[mask, "rating_denominator"] = denominator


# In[554]:


# 9/11
this_id = GetTweetId(archive_master_clean, 9, 11)
PrintTextRatings(archive_master_clean, this_id)


# In[555]:


ResetRatings(archive_master_clean, this_id, 14, 10)


# ###### Test

# In[556]:


PrintTextRatings(archive_master_clean, this_id)


# ###### Code

# In[557]:


# 11/15
sum(archive_master_clean.rating_denominator == 15)


# Looks like this record has been removed since the time we performed the assessment. Hence, we don't need to address this.

# ###### Code

# In[558]:


# 960/0
sum(archive_master_clean.rating_denominator == 0)


# Looks like this record has been removed since the time we performed the assessment. Hence, we don't need to address this.

# ###### Code

# In[559]:


sum(archive_master_clean.rating_denominator == 20)


# In[560]:


# 4/20
this_id = GetTweetId(archive_master_clean, 4, 20)
PrintTextRatings(archive_master_clean, this_id)


# In[561]:


ResetRatings(archive_master_clean, this_id, 13, 10)


# ###### Test

# In[562]:


PrintTextRatings(archive_master_clean, this_id)


# ###### Code

# In[563]:


# 50/50
sum(archive_master_clean.rating_denominator == 50)


# In[564]:


this_id = GetTweetId(archive_master_clean, 50, 50)
PrintTextRatings(archive_master_clean, this_id)


# In[565]:


ResetRatings(archive_master_clean, this_id, 11, 10)


# ###### Test

# In[566]:


PrintTextRatings(archive_master_clean, this_id)


# ###### Code

# In[567]:


# 1/2
sum(archive_master_clean.rating_denominator == 2)


# In[568]:


this_id = GetTweetId(archive_master_clean, 1, 2)
PrintTextRatings(archive_master_clean, this_id)


# In[569]:


ResetRatings(archive_master_clean, this_id, 9, 10)


# ###### Test

# In[570]:


PrintTextRatings(archive_master_clean, this_id)


# ##### `image_predictions` Table: Breed names have "_" instead of " "
# ###### Define
# Replace "-" with " " in each of the "p1", "p2", and "p3" columns using series.str.replace() method
# ###### Code

# In[571]:


archive_master_clean.loc[:, "p1"] = archive_master_clean.p1.str.replace("_", " ")
archive_master_clean.loc[:, "p2"] = archive_master_clean.p2.str.replace("_", " ")
archive_master_clean.loc[:, "p3"] = archive_master_clean.p3.str.replace("_", " ")


# ###### Test

# In[572]:


archive_master_clean.p1.value_counts().head(10)


# In[573]:


archive_master_clean.p2.value_counts().head(10)


# In[574]:


archive_master_clean.p3.value_counts().head(10)


# "-" should be replaced with " "

# ##### `image_predictions` Table : Breed names sometimes lowercase, sometimes upper case.
# ###### Define
# Use series.str.title() to make all breed names title case.
# ###### Code

# In[575]:


archive_master_clean.loc[:, "p1"] = archive_master_clean.p1.str.title()
archive_master_clean.loc[:, "p2"] = archive_master_clean.p2.str.title()
archive_master_clean.loc[:, "p3"] = archive_master_clean.p3.str.title()


# ###### Test

# In[576]:


archive_master_clean.p1.value_counts().head(10)


# In[577]:


archive_master_clean.p2.value_counts().head(10)


# In[578]:


archive_master_clean.p3.value_counts().head(10)


# All breed names should be title case as in "This Is Title Case"

# ##### `archive` table : Many ratings are not for dogs
# ###### Define
# Some tweets humorously rate animals other than dogs, typically with low ratings such as 3/10.
# We do not want to remove these tweets from the table as they could be useful in some analysis. However we can add two columns as follows:
# 
# * add an "is_likely_dog" column if there is at least one image prediction claiming the image has a dog with a confidence of at least 20%. boolean data type
# * Add a "likely_breed" column for the breed with the larget confidence and where the "is_likely_dog" is True

# ###### Code

# In[579]:


def LikelyDog(row):
    conf_threshold = 0.20
    # Default values
    is_likely_dog = False
    likely_breed = np.nan
    
    # COnstruct DF
    df_list = {"conf" : [row.p1_conf, row.p2_conf, row.p3_conf],
               "is_dog" : [row.p1_dog, row.p2_dog, row.p3_dog],
               "breed" : [row.p1, row.p2, row.p3]}
    df = pd.DataFrame(df_list)
    
    # Remove rows that are not a dog
    mask = (df.is_dog == True)
    df = df[mask]
    # Select rows based on a minimum confidence threshold
    mask = (df.conf >= conf_threshold)
    df = df[mask]
    # Sort by confidence, descending
    df = df.sort_values(by = ["conf"], ascending = False)
    
    if (len(df) >=1):
        is_likely_dog = True
        likely_breed = df.breed.iloc[0]
    
    
    #return(df)
    return pd.Series([row.tweet_id, is_likely_dog, likely_breed], index=["tweet_id", "is_likely_dog","likely_breed"])
    


# In[580]:


dog_info = archive_master_clean.apply(LikelyDog, axis = 1)
dog_info.head()


# In[581]:


archive_master_clean = archive_master_clean.merge(right = dog_info, on = "tweet_id", how = "left")


# ###### Test

# In[582]:


archive_master_clean.info()


# The table should have 27 columns including the "is_likely_dog" and "likely_breed" columns

# In[583]:


def ImagePrediction_DF(row):
    conf_threshold = 0.01
    df_list = {"conf" : [row.p1_conf, row.p2_conf, row.p3_conf],
               "is_dog" : [row.p1_dog, row.p2_dog, row.p3_dog],
               "breed" : [row.p1, row.p2, row.p3]}
    df = pd.DataFrame(df_list)
        
    is_likely_dog = True
    likely_breed = "Test"
    return(df)


# In[584]:


ImagePrediction_DF(archive_master_clean.iloc[5])


# In[585]:


LikelyDog(archive_master_clean.iloc[5])


# The likely dog breed from the "LikelyDog" method should match that which is visually identified from the output of the previous cell

# ## Data Storage
# The data has been wrangled. The final cleaned table `archive_master_clean` will be stored in a Comma Separated Values (CSV) File .

# In[586]:


file_name = "twitter_archive_master.csv"
archive_master_clean.to_csv(file_name)


# ## Analysis and Visualization

# #### Ratings Distribution
# I am curious to see the distribution of ratings

# In[587]:


archive_master_clean.rating_denominator.value_counts().head(8)


# In[588]:


sum(archive_master_clean.rating_denominator == 10)/sum(archive_master_clean.rating_denominator.notnull())


# Most of the ratings, about 99.4%, have a denominator of 10.

# In[589]:


archive_master_clean.rating_numerator.value_counts().head(8)


# In[590]:


ratings = archive_master_clean.rating_numerator.astype(str) + "/" + archive_master_clean.rating_denominator.astype(str)
ratings.value_counts().head(8)


# Let us plot the distribution of the numerator

# In[591]:


fig = plt.figure(figsize = (6, 6))
ax = plt.gca()
mask = (archive_master_clean.rating_denominator == 10)

archive_master_clean.loc[mask, "rating_numerator"].value_counts().sort_index().plot.bar(ax = ax)
ax.set_xlim([0,15])
ax.set_title("Distribution of Rating Numerator")
ax.set_xlabel("Rating Numerator")
ax.set_ylabel("Count")


# The most common rating is 12/10.
# Based on looking at the tweets with the large numerators, the numerators larger than 14 are for pictures with more thana 1 dog. Hence we can say that 14/10 is the best rating given to any dog. 
# 
# Let us filter out tweets that may not be about dogs using the "is_likely_dog" column that was derived from the image predictions.

# In[592]:


fig = plt.figure(figsize = (6, 6))
ax = plt.gca()
mask = (archive_master_clean.rating_denominator == 10)
archive_master_clean.loc[mask, "rating_numerator"].value_counts().sort_index().plot.line(ax = ax, label = "All Tweets")
ax.set_title("Distribution of Rating Numerator")
ax.set_xlabel("Rating Numerator")
ax.set_ylabel("Count")

mask = mask & (archive_master_clean.is_likely_dog)
archive_master_clean.loc[mask, "rating_numerator"].value_counts().sort_index().plot.line(ax = ax, color = 'g', label = "Tweets Likely of Dogs")
ax.set_xlim([0,15])
ax.legend()


# Looks like the "is_likely_dog" filter is removing records across all ratings. This was not expected. The original hypothesis was that the ratings with the low numerators were for non-dog ratings. Though the above filter does reduce the number of records with low ratings, it also reduces the number of records with more "approriate-for-dogs" ratings. This feature probably needs rethinking.

# #### Tweets-Rate Over Time
# 
# I am curious to see the rate at which the account was tweet and the variation of this quantity over time.

# In[593]:


start_date = archive_master_clean.timestamp.sort_values().iloc[0]
end_date = archive_master_clean.timestamp.sort_values().iloc[-1]
print("Tweets go from {} through {}".format(start_date.strftime("%b %d %Y"), end_date.strftime("%b %d %Y")))


# In[594]:


timestamp = archive_master_clean.loc[:, ["timestamp","tweet_id"]].set_index("timestamp")
daily_rate = timestamp.groupby(pd.TimeGrouper(freq = "D"))["tweet_id"].count()
smoothed = daily_rate.groupby(pd.TimeGrouper(freq = "10D")).mean()
smoothed.head()


# In[595]:


smoothed.tail()


# In[596]:


# Plot
fig = plt.figure(figsize = (10,6))
ax = fig.gca()

smoothed.plot()
ax.set_ylim(bottom = 0)
ax.set_xlabel("")
ax.set_ylabel("Tweet Rate (Tweets per Day)")
ax.set_title("Variation of Tweet Rate over Time")


# We can see that the tweet rate drop greatly from ~18 tweets per day during November 2015 to around 1.6 tweets per day during July 2017. Note this done not include retweets by @dog_rates.

# #### Variation of Ratings with Time
# 
# Again we will only consider ratings wth a denominator of 10

# In[597]:


mask = (archive_master_clean.rating_denominator == 10)
rating = archive_master_clean.loc[mask, ["timestamp", "rating_numerator"]].set_index("timestamp")
rating_mean = rating.groupby(pd.TimeGrouper(freq = "10d"))["rating_numerator"].mean()
rating_mean.head()


# In[598]:


# Plot
fig = plt.figure(figsize = (10,6))
ax = fig.gca()

rating_mean.plot(ax = ax)
ax.set_ylim(bottom = 0)
ax.set_xlabel("")
ax.set_ylabel("Rating Numerator, 10 day Mean")
ax.set_title("Variation of Rating Numerator over Time")


# Let us investigate the large spike near July 2016 

# In[599]:


rating.info()


# In[600]:


rating.loc[datetime.date(year = 2016, month = 8, day = 1):
           datetime.date(year = 2016, month = 7, day = 1)].sort_values(by = "rating_numerator", ascending = False).head()


# In[601]:


printOriginal_RatingFilter(1776)


# We can see that the tweet from "2016-07-04 15:00:45" has a rating of 1776/10. This humorous rating about the dog being "simply America af" is throwing off our mean rating. However, this is not an error in the cleaning as this is an actual rating. We shall remove this tweet ad-hoc for the purpose of this chart alone.

# In[602]:


mask = (archive_master_clean.rating_denominator == 10)
#remove tweet with rating of 1776/10
mask = mask & (archive_master_clean.rating_numerator != 1776)

rating = archive_master_clean.loc[mask, ["timestamp", "rating_numerator"]].set_index("timestamp")
rating_mean = rating.groupby(pd.TimeGrouper(freq = "M"))["rating_numerator"].mean()
rating_mean.head()

# Plot
fig = plt.figure(figsize = (10,6))
ax = fig.gca()

rating_mean.plot(ax = ax)
#ax.set_ylim(bottom = 0)
ax.set_xlabel("")
ax.set_ylabel("Rating Numerator, Monthly Mean")
ax.set_title("Variation of Rating Numerator over Time")


# A clear trend of the average ratings increasing over time is seen in the chart above.

# #### Tweet Popularity over Time
# 
# I am curious to see how popular @dog_rates's tweets were with the twitter audience over time. I will define a tweet's "popularity" as the sum of the favorite count and the retweet count for that tweet. This is a simple metric and does not account appropriately for situations such as the same twitter user both retweeting and favoriting a tweet.

# In[603]:


engagement = archive_master_clean.loc[:, ["timestamp", "retweet_count", "favorite_count"]].copy().set_index("timestamp")

engagement["popularity"] = engagement.retweet_count + engagement.favorite_count
engagement["retweet_to_favorite"] = engagement.retweet_count / engagement.favorite_count
engagement.head()


# In[604]:


smooth_freq = "M"
engagement_mean = engagement.groupby(pd.TimeGrouper(freq = smooth_freq)).mean()
engagement_mean.head()


# In[605]:


engagement_mean.tail()


# In[606]:


# Plot
fig = plt.figure(figsize = (10,6))
ax = fig.gca()

engagement_mean.plot(y = "popularity", ax = ax)
#ax.set_ylim(bottom = 0)
ax.set_xlabel("")
ax.set_ylabel("Tweet Popularity (Retweets and Favorites per tweet), Monthly Mean")
ax.set_title("Tweet Popularity over Time")
ax.legend_.remove()


# We can see that the popularity of the tweets has grown tremendously over time, from ~1,800  retweets and likes per tweet in Nov 2015 to ~43,000 in July 2017.

# #### Most Popular Breeds
# 
# Let us plot the average popularity of different dog breeds. I will only look at breeds which appear in 20 or more tweets.

# In[607]:


archive_master_clean.info()


# In[608]:


sum(archive_master_clean.likely_breed.value_counts() > 20)


# In[609]:


engagement = archive_master_clean.loc[:, ["timestamp", "retweet_count", "favorite_count", "likely_breed", "jpg_url", "text"]].copy().set_index("timestamp")
engagement["popularity"] = engagement.retweet_count + engagement.favorite_count
engagement.head()


# In[610]:


# only keep tweets of breeds that 20 or more tweets
engagement = engagement.groupby("likely_breed").filter(lambda x: len(x) > 20)

breed_mean = engagement.groupby("likely_breed").mean().sort_values("popularity", ascending = False)
breed_mean.head()


# In[611]:


# Plot
fig = plt.figure(figsize = (10,6))
ax = fig.gca()

breed_mean.plot.bar(y = "popularity", ax = ax)
#ax.set_ylim(bottom = 0)
ax.set_xlabel("")
ax.set_ylabel("Average Popularity (Retweets and Favorites per tweet)")
ax.set_title("Popularity of Different Breeds")
ax.legend_.remove()


# Note that this plot does not account for the drastic increase in tweet popularity over time. This chart would be more accurate if this effect were corrected.

# In[612]:


mask = (engagement.likely_breed == "French Bulldog")
engagement.loc[mask].sort_values("popularity", ascending = False).iloc[0]


# In[613]:


engagement.loc[mask].sort_values("popularity", ascending = False).iloc[0].loc["text"]

