# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 21:49:37 2024

@author: Administrator
"""

import gzip
from collections import defaultdict, Counter
import random
import pandas as pd
import os
import json
import subprocess
from tqdm import tqdm
import itertools
import numpy as np
import csv

import re
from geotext import GeoText
import pycountry


openalexPath = 'OpenAlex_20240227/openalex-snapshot-tsv-files'

df_country_list_different_spellings=pd.read_excel('country_list_different_spellings.xlsx')
df_country_list_different_spellings['ulke']=df_country_list_different_spellings['ulke'].apply(lambda x: x.lower())
map_country_list_different_spellings = pd.Series( df_country_list_different_spellings.ulke.values, index=df_country_list_different_spellings.yer).to_dict()
map_country_list_different_spellings['Turkey']='tur'

def inverted_index_to_text(inverted_index_str):
    try:
        # Remove the outermost quotes and unescape the inner quotes
        inverted_index_str = inverted_index_str[1:-1]
        inverted_index_str = inverted_index_str.replace('""', '"')
        
        # Now parse the JSON
        try:
            inverted_index = json.loads(inverted_index_str)
        except json.JSONDecodeError:
            raise ValueError("Unable to parse the input string. Please check the format.")
        
        # Create a list of (position, word) tuples
        word_positions = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))
        
        # Sort the list by position
        word_positions.sort()
        
        # Join the words to get the original text
        original_text = ' '.join(word for _, word in word_positions)
    
        return original_text
    
    except:
        return ''

def get_country(title, abstract):
    # Assuming title is a list of dictionaries with 'title' as a key, and abstract is a string.
    txt = title+' '+ abstract

    for copyright_mark in ['漏', 'Copyright (C)']:
        if copyright_mark in txt:
            txt = txt.split(copyright_mark)[0]

    for tag in [ 'US dollar','New Mexico','Turkish','US$','US $','United States Dollar','USD','HK', 'Congo Red',
               'Congo red', 'US-Dollar']:
        if tag in txt:
            txt=txt.replace(tag, '')

    places = GeoText(txt)
    country_codes = set()  # To store unique country codes.

    # Attempt to get country codes directly from GeoText results.
    for country_name in set(places.countries):
        country_code = pycountry.countries.get(name=country_name)
        if country_code:
            country_codes.add(country_code.alpha_3.lower())
        else:
            # Directly add to undefined list if not found.
            country_codes.add(map_country_list_different_spellings.get(country_name, country_name))

    pattern = r'\b(' + '|'.join(re.escape(country_name) for country_name in map_country_list_different_spellings.keys()) + r')\b'

    # Find all occurrences of country names in the text, considering word boundaries.
    matches = re.findall(pattern, txt)

    # For each matched country name, add its corresponding country code to the set.
    for match in matches:
        country_codes.add(map_country_list_different_spellings[match])


    # Handle special case for 'uae'.
    if 'uae' in country_codes:
        country_codes.remove('uae')
        country_codes.add('are')

    return list(country_codes)


usecols = ['id', 'title','publication_year', 'type', 'cited_by_count', 'language', 'abstract_inverted_index', 'is_paratext', 'doi']

paper_mentions = {}
paper_year = defaultdict(int)
paper_title = {}
paper_abstract = {}

# LOAD id
chunksize = 100000
_result = subprocess.run(['wc', '-l', os.path.join(openalexPath, 'works.tsv')], capture_output=True, text=True)
total_lines = int(_result.stdout.split()[0])
total_chunks = total_lines // chunksize + (1 if total_lines % chunksize != 0 else 0)

for chunk in tqdm(pd.read_csv(os.path.join(openalexPath, 'works.tsv'),
                              chunksize=chunksize, 
                              sep="\t", 
                              encoding='utf-8', 
                              low_memory=False,
                              quoting=csv.QUOTE_NONE,
                              usecols=usecols),
                  total=total_chunks,  # Estimate total iterations
                  desc="Processing chunks"):
     
    chunk = chunk.copy()
    chunk = chunk.dropna(subset=usecols)
    chunk['cited_by_count'] = pd.to_numeric(chunk['cited_by_count'], errors='coerce')
    chunk['publication_year'] = pd.to_numeric(chunk['publication_year'], errors='coerce')
    chunk['is_paratext'] = chunk['is_paratext'].astype(str).str.lower() == 'true'
    chunk = chunk[
                    (chunk['publication_year'].between(2003, 2018)) &
                    (chunk['cited_by_count'] > 0) &
                    (chunk['language'] == 'en') &
                    (chunk['type'] == 'article') &
                    (chunk['abstract_inverted_index'].str.len() > 100)
                 ]
    
    if chunk.shape[0]>0:
        chunk['abstract'] = chunk['abstract_inverted_index'].apply(inverted_index_to_text)
        chunk = chunk[chunk['abstract'].str.len() < 3000] # avoid long abstracts which are normally editorial note
        chunk['country_detected'] = chunk.apply(lambda x: get_country(x['title'], x['abstract']), axis=1)
        # get values
        paper_mentions.update(dict(zip(chunk['id'], chunk['country_detected'])))
        paper_year.update(dict(zip(chunk['id'], chunk['publication_year'])))

'''
#get samples
'''
# get topic to field
topics_df = pd.read_csv(os.path.join(openalexPath, 'topics.tsv'), sep='\t')
topicid_field = dict(zip(topics_df['id'], topics_df['field_display_name']))

#get paper to topic
select_paper_set = set(paper_year.keys())
paper_field = defaultdict(lambda: ['', 0])
field_set = set()

# Count total lines for tqdm
_result = subprocess.run(['wc', '-l', os.path.join(openalexPath, 'works_topics.tsv')], capture_output=True, text=True)
total_lines = int(_result.stdout.split()[0])
rows_processed = 0
update_frequency = 10000000

with open(os.path.join(openalexPath, 'works_topics.tsv'), 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)  # Skip header row
    pbar = tqdm(total=total_lines, desc='Processing works_topics.tsv', unit='row')
    for i, row in enumerate(reader, start=1):
        rows_processed += 1
        work_id, topic_id, score = row[0], row[1], float(row[2])
        if work_id in select_paper_set and score > paper_field[work_id][1]:
            paper_field[work_id] = [topicid_field[topic_id], score]
            field_set.add(topicid_field[topic_id])
        if rows_processed % update_frequency == 0:
            pbar.update(update_frequency)
    pbar.close()
print ('paper field relationships read in.....')
print (len(paper_field), 'paper field relationships are loaded...')

# get 20 papers from each field
sample_20 = []
for f in field_set:
    _f_papers = [_p for _p, [_f, _s] in paper_field.items() if _f==f and _s!=0]
    sample_20 += random.sample(_f_papers, 20)
# get 280 papers from those with country mention
sample_mention = random.sample([p for p,v in paper_mentions.items() if p not in sample_20 and v!=[]], 280)
# get 200 papers from those with specific country mention
select_country = ['egy', 'tun', 'lby', 'syr', 'yem', 'bhr', 'jor', 'kwt', 'mar', 'omn']
specific_plist = [_p for _p, _m in paper_mentions.items() if set(_m) & set(select_country)]
# Get 20 papers for each country in the list
sample_specific = [
    p for country in select_country
    for p in random.sample(
        [p for p, m in paper_mentions.items() if country in m and p not in sample_20 and p not in sample_mention],
        min(20, len([p for p, m in paper_mentions.items() if country in m and p not in sample_20 and p not in sample_mention]))
    )
]
print ('samples obtained...')
print (len(sample_20), len(sample_mention), len(sample_specific))
'''
#load work again just to get title and abstract
'''
all_sample_id = sample_20+sample_mention+sample_specific
print (len(all_sample_id), 'len_all_sample_id')
for chunk in pd.read_csv(os.path.join(openalexPath, 'works.tsv'),
                              chunksize=chunksize, 
                              sep="\t", 
                              encoding='utf-8', 
                              low_memory=False,
                              quoting=csv.QUOTE_NONE,
                              usecols=usecols):
     
    chunk = chunk[chunk['id'].isin(all_sample_id)]

    if chunk.shape[0]>0:
        chunk['abstract'] = chunk['abstract_inverted_index'].apply(inverted_index_to_text)
        paper_title.update(dict(zip(chunk['id'], chunk['title'])))
        paper_abstract.update(dict(zip(chunk['id'], chunk['abstract'])))
print ('abstract title dict len', len(paper_title), len(paper_abstract))

'''
#save data
'''
labels = ['field_20', 'with_mention', 'with_mention_arab']
data = []
for i, ids in enumerate([sample_20, sample_mention, sample_specific]):
    for _id in ids:
        data.append({
            'ID': _id,
            'Year':paper_year.get(_id, 0),
            'Field':paper_field.get(_id,['',0])[0],
            'SampleGroup': labels[i],
            'Mentions':paper_mentions.get(_id,[]),
            'Title': paper_title.get(_id, '').replace('\n', ' ').strip(),
            'Abstract': paper_abstract.get(_id, '').replace('\n', ' ').strip(),
            
        })
df = pd.DataFrame(data)
print (df.shape, 'df.shape')
df.to_excel('samples.xlsx', index=False)
print ('final results saved!')
