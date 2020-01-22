import os
import csv
import linecache as lc
import tweepy as tw # Installed via pip
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import Regexptokenizer
from nltk.stem import WordNetLemmatizer

# 1) Create dataframe with pandas - use the text only csv?
# 2) Can single out the tweet text column
# 3) Remove RTs, name mentions (@s), punctation
# 4) Remove hashtags and space out their components?
# 5) Change ampersands to 'ands' and reduce newlines to spaces
# 6) lowercase all text
# 7) Tokenise and pad text
