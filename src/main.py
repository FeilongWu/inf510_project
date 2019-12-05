import sqlite3 as sql
import pandas as pd
import numpy as np
from sqlcm import select_statement
'''
This program will run statistical analysis and generate graphs to understand
and answer questions about some demographic variables whose values are stored
in a local database.
'''

database='..//data//Demography.db'

def attr_names(db,table_name):
    # returns a list of attribute names in the database table
    # db = connection to the database
    # table_name=the name of database table
    cursor=db.execute(f'SELECT * FROM {table_name}')
    return list(map(lambda x: x[0], cursor.description))
def main():
    db=sql.connect(database)
    cur=db.cursor()
    table_name_yr='Year'
    table_name_st='State_abbreviation'
    table_name_ct='County'
###----------------find counties with fastest growing population-----------###
    table_name='Birth_rate'
    attributes=attr_names(db,table_name) # get the list of column names
    attr_ct,attr_br='county','birth_rate'
    # find the index of a column
    county_idx,br_idx=attributes.index(attr_ct),attributes.index(attr_br) 
    cur.execute(*select_statement(table_name,[attr_ct]))
    county=[]
    for i in cur.fetchall():
        # i = a list of all values in a row: [id, county, birth rate...]
        county.append(i[county_idx])
    county=list(set(county)) # remove duplicates
    start_yr,end_yr=2016,2018
    birth_rates={} # {county1:[rate yr1,rate yr2...]...}
    for i in county:
        birth_rates[i]=[]
    for i in range(start_yr,end_yr+1):
        for j in birth_rates:
            cur.execute(*select_statement(table_name,[attr_ct,'year'],j,i,select=[attr_br]))
            try:
                birth_rates[j].append(cur.fetchone()[0])
            except:
                birth_rates[j].append(None)
    for i in birth_rates:
        if None in birth_rates[i]:
            birth_rates.pop(i) # remove county with None value
        else:
            birth_rates[i]=sum(birth_rates[i])/len(birth_rates[i]) # calculate the mean
    birth_county=[] # takes the form [(birth rate1,county1),...]
    for ct,br in birth_rates.items():
        birth_county.append((br,ct))
    birth_county=sorted(birth_county)
    num=5 # the number of counties
    county_num=[]# store the foreign keys
    birth_county.reverse()
    for i in birth_county[0:num]:
        county_num.append(i[1])
    county_state=[] # store county and state names
    for i in county_num:
        cur.execute(*select_statement(table_name_ct,['id'],i,select=[attr_ct,'state']))
    print(len(birth_rates))
if __name__=='__main__':
    main()
