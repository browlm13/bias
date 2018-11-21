#!/usr/bin/env python3

"""
	
	Load article headlines and sources into pandas DataFrame.

			columns = ['title', 'source']

	example usage:

		from headlines import article_headlines

		# selected sources
		ex_sources = ['Fox News', 'MSNBC', 'Reuters']

		# load into pandas dataframe
		selected_news_df = article_headlines(ex_sources)

"""

__author__  = "LJ Brown"
__file__ = "headlines.py"

# imports

# external
import pandas as pd

def _get_news_df():

	# for source bellow
	# read in titles and publishers from news archieve
	SOURCE = '../data/uci-news-aggregator.csv'

	# read in titles and publishers from news archieve
	df = pd.read_csv(SOURCE)
	news_df = df[['TITLE', 'PUBLISHER']]

	# rename column headers to standard format
	# title, source
	news_df.rename(columns={'TITLE':'title','PUBLISHER':'source'}, inplace=True)

	return news_df

def article_headlines(selected_sources):

	# load complete news archieve into dataframe
	news_df = _get_news_df()

	# limit to selected sources
	selected_df = news_df.loc[news_df['source'].isin(selected_sources)]

	return selected_df

# example usage
if __name__ == '__main__':

	# selected sources
	ex_sources = ['Fox News', 'MSNBC', 'Reuters']

	# load into pandas dataframe
	selected_news_df = article_headlines(ex_sources)

	# display dataframe
	print(selected_news_df)


