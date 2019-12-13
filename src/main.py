import sqlite3 as sql
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import copy
from scipy.optimize import curve_fit
from sqlcm import select_statement
from mpl_toolkits.mplot3d import Axes3D
'''
This program will run statistical analysis and generate graphs to understand
and answer questions about some demographic variables whose values are stored
in a local database.
'''



def attr_names(db,table_name):
    # returns a list of attribute names in the database table
    # db = connection to the database
    # table_name=the name of database table
    cursor=db.execute(f'SELECT * FROM {table_name}')
    return list(map(lambda x: x[0], cursor.description))
def get_county_state(database,county_id):
    # returns string of COUNTY, STATE
    # database=database name
    # table=table name
    # county_id=a list of county primary keys
    db=sql.connect(database)
    cur=db.cursor()
    county_state=[] # store county and state names
    for i in county_id:
        cur.execute(*select_statement(table_name_ct,['id'],i,select=['county','state']))
        temp=cur.fetchone()
        county_name,state_num=temp[0],temp[1]
        county_name=county_name[0].upper()+county_name[1:] # capitalize initial
        cur.execute(*select_statement(table_name_st,['id'],state_num,select=['state']))
        temp=cur.fetchone()[0]
        county_state.append(county_name+', '+temp[0].upper()+temp[1:])
    return county_state
def get_year_specific(db,cur,years,table_name,attr,join_attr,group_attr='year'):
    # returns a data frame modified from a datasebase table
    # the modification is that the values in a target column, which are
    # aggregated on multiple years, are extracted for each target year
    # and joint as a new dataframe
    # db=database connection
    # cur=cursor of a database
    # years=a list of target years
    # table_name=the name of database table
    # attr=target attribute whose values are aggregated on multiple years
    # join_attr=the attribute name on which the data frames will join
    # group_attr=the attribute by which the values will group (now just consider
    # year)
    attr_id='id'
    new_cow={} # new column names
    cur.execute(*select_statement(table_name,[attr]))
    attributes=attr_names(db,table_name)
    df=pd.DataFrame(cur.fetchall())
    for i in range(len(attributes)):
        new_cow[i]=attributes[i]
    df=df.rename(columns=new_cow)
    try:
        df=df.drop([attr_id],axis=1)
    except KeyError:
        pass
    df=df.groupby(group_attr)
    df_years=[] # store data frames gourped by year
    for i in years:
        try:
            temp=df.get_group(i)
        except KeyError:
            continue
        temp=temp.rename(columns={attr:attr+str(i)})
        temp=temp.drop([group_attr],axis=1)
        df_years.append(temp)
    if len(df_years)==1:
        joint=df_years[0]
    else:
        joint=df_years[0]
        joint=joint.set_index(join_attr)
        for i in df_years[1:]:
            joint=joint.join(i.set_index(join_attr))
    return joint
def normalize(array):
    # return a normalized array preserving None and NaN, which are excluded
    # from calculation
    # array = numpy array
    new_array=[]
    
    for i in array:
        if type(i)==int:
            new_array.append(i)
        elif type(i)==float:
            if math.isnan(i):
                continue
            else:
                new_array.append(i)
        else:
            try:
                i=float(i)
                new_array.append(i)
            except:
                continue
    
    min_val=min(new_array)
    dif=max(new_array)-min_val
    result=[]
    if dif==0:
        entry=1
    else:
        entry=0
    if entry:
        for i in range(len(array)):
            if type(array[i])==int:
                result.append(entry)
            elif type(array[i])==float:
                if math.isnan(array[i]):
                    result.append(array[i])
                    continue
                else:
                    result.append(entry)
            else:
                try:
                    array[i]=float(array[i])
                    result.append(entry)
                except:
                    result.append(array[i])
                    continue
    else:
        
        for i in range(len(array)):
            if type(array[i])==int:
                result.append((array[i]-min_val)/dif)
            elif type(array[i])==float:
                if math.isnan(array[i]):
                    result.append(array[i])
                    continue
                else:
                    result.append((array[i]-min_val)/dif)
            else:
                try:
                    temp=float(array[i])
                    result.append((array[i]-min_val)/dif)
                except:
                    result.append(array[i])
                    continue
    
    return np.array(result)
def normalize_all(data):
    # return normalized lists in a list
    # data=a list of lists containing values
    for i in range(len(data)):
        data[i]=normalize(data[i])
    return data
def normalize_df(df,columns='All'):
    # returns the df whose columns are normalized and specified by columns
    # df=pandas data frame
    # columns=a list of names of the columns to be normalized
    if type(columns)== str and columns.lower()=='all':
        columns=list(df.columns)
    full_col=list(df.columns)
    for i in columns:
        df[i]=normalize(df[i].to_numpy())
    drop_list=[]
    for i in full_col:
        if i in columns:
            continue
        else:
            drop_list.append(i)
    if len(drop_list)==0:
        return df
    else:
        return df.drop(columns=drop_list)
def extract_features(df,years,order):
    # returns a list of values extracted from input dataframe
    # return list looks like [[values of attr1],[values of attr2]...]
    # df=pandas data frame
    # years=a list of years
    # order=a list of attribute names in an order
    result=[]
    for i in range(len(order)):
        result.append([])
    columns=list(df.columns)
    valid_col=[]
    for i in years:
        temp=[]
        for j in order:
            col=j+str(i)
            if col in columns:
                temp.append(col)
            else:
                break
        valid_col.append(temp)
    for i in valid_col:
        for j in range(len(i)):
            result[j].extend(list(df[i[j]].to_numpy()))
    return result
def clean_val(values):
    # remove NaN, inf, etc. in values
    # values = a list of list(s) containing some values
    non_val=set()
    for i in values:
        for j in range(len(i)):
            if math.isnan(i[j]) or i[j]==None or math.isinf(i[j]):
                non_val.add(j)
    non_val=list(non_val)
    non_val.sort(reverse=True)
    for i in non_val:
        for j in range(len(values)):
            del values[j][i]
def avg_birth_rate(table_name,db,cur,df,start_yr,end_yr):
    # return a dictionary of the form: {county key:average birth rate,...}
    # find the index of a column
    # table_name=name of database table
    # db=database connection
    # cur=database cursor
    # df=dataframe of birth rate
    # start_yr=start year
    # end_yr=end year
    attributes=attr_names(db,table_name) # get the list of column names
    county_idx,br_idx=attributes.index(attr_ct),attributes.index(attr_br)
    cur.execute(*select_statement(table_name,[attr_ct]))
    county=[]
    for i in cur.fetchall():
        # i = a list of all values in a row: [id, county, birth rate...]
        county.append(i[county_idx])
    county=list(set(county)) # remove duplicates
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
    sorted(birth_county)
    birth_county.reverse()
    return birth_county
def get_county_key(num,birth_county):
    # return a list of county keys from a list
    # num=the number of keys
    # birth_county=[[birth rates],[county keys]]
    county_num=[]# store the foreign keys
    for i in birth_county[0:num]:
        county_num.append(i[1])
    return county_num
def join_df(df1,df2):
    # return a joint data frame from the inputs
    # df1=a data frame
    # df2=nother data drame
    return df1.join(df2)
def extract_array(df,column_order,attr):
    # return an array for data extract from a data frame
    # df=data frame
    # column_order=a list of column names for the data frame
    # attr=a list of the names of columns for extraction
    result=[]
    for i in attr:
        result.append(np.array(df[column_order.index(i)]))
    return result
def get_2D_array(x,y,func,p1,num=200):
    # return x,y,z as 2D arrays for surface plot
    # x=1D array
    # y=1D array
    # func=anonumous function to generate z
    # p1=parameters for func
    # num=the number of generated points
    X,Y=np.meshgrid(np.linspace(min(x),max(x),num),\
                                 np.linspace(min(y),max(y),num))
    Z=func((X.ravel(),Y.ravel()),*p1[0]).reshape(X.shape)
    return X,Y,Z
        
def main():

    
    
    
###----------------find counties with fastest growing population-----------###
    table_name='Birth_rate'
    years=[2013,2016,2017,2018]
    start_yr,end_yr=2016,2018
    num=5 # the number of counties
    birth_rate=get_year_specific(db,cur,years,table_name,attr_br,attr_ct)
    birth_county=avg_birth_rate(table_name,db,cur,birth_rate,start_yr,end_yr)
    county_num=get_county_key(num,birth_county)   
    county_state=get_county_state(database,county_num)
    print(f'The {num} state(s) with the fastest growing population is(are):')
    for i in county_state:
        print(i)

###--------------Analyze Income per Capita and Birth Rate-------------------###
    table_name='Income_per_capita'
    attr_income='per_capita_income'
    joint_income=get_year_specific(db,cur,years,table_name,attr_income,attr_ct)
    # join both income and birth rate data frames
    joint_income_birth=join_df(joint_income,birth_rate)
    # make a copy before normalization
    joint_income_birth_copy=copy.copy(joint_income_birth)
    #joint_income_birth=normalize_df(joint_income_birth)
    extract_income_birth=extract_features(joint_income_birth,years,[attr_br,attr_income])
    clean_val(extract_income_birth)
    # correlation coefficient
    extract_income_birth=normalize_all(extract_income_birth)
    corr_income_birth=np.corrcoef(extract_income_birth[0],extract_income_birth[1])[0][1]
    plt.figure(1)
    plt.scatter(extract_income_birth[0],extract_income_birth[1])
    plt.xlabel('Normalized Birth Rate')
    plt.ylabel('Normalized Income per Capita')
    plt.title('Income per Capita vs. Birth Rate for 2016-18')
    plt.text(0.2,0.7,f'correlation coefficient = {corr_income_birth}')
    
###---------------Analyze Income per Capita and Unemployment Rate------------###
    table_name='Unemployment'
    attr_ep='unemployment_rate'
    joint_unemploy=get_year_specific(db,cur,years,table_name,attr_ep,attr_ct)
    joint_income_unemploy=join_df(joint_income,joint_unemploy)
    extract_income_unemploy=extract_features(joint_income_unemploy,years,[attr_income,attr_ep])
    clean_val(extract_income_unemploy)
    # correlation coefficient
    corr_income_unemploy=np.corrcoef(extract_income_unemploy[0],extract_income_unemploy[1])[0][1]
    func=lambda x,a,b: a*x+b # linear regression line
    a,b = np.polyfit(extract_income_unemploy[0],extract_income_unemploy[1], 1)
    plt.figure(2)
    #plt.scatter(extract_income_unemploy[0],extract_income_unemploy[1])
    plt.plot(extract_income_unemploy[0],extract_income_unemploy[1],'o',extract_income_unemploy[0],\
        list(func(np.array(extract_income_unemploy[0]),a,b)),'-r')
    plt.ylim(0,max(extract_income_unemploy[1])+1)
    plt.xlabel('Income per Capita ($)')
    plt.ylabel('Unemployment Rate (%)')
    plt.title('Unemployment Rate vs. Income per Capita for 2013 and 2016-18')
    plt.text(7e04,22,f'correlation coefficient = {corr_income_unemploy}')
    
###---------------Analyze Unemployment, Birth Rate, and Home Price-----------###
    table_name='Home_price'
    column_order=[attr_br,attr_income,attr_pr]
    joint_price=get_year_specific(db,cur,years,table_name,attr_pr,attr_ct)
    joint_income_birth_price=join_df(joint_income_birth_copy,joint_price)
    extract_income_birth_price=extract_features(joint_income_birth_price,years,\
                                          column_order)
    clean_val(extract_income_birth_price)
    # fit this anonymous function:
    # birth rate = a*income**2+b*income+c*home price**2+d*home prince+e
    func=lambda xy,a,b,c,d,e:a*xy[0]**2+b*xy[0]+c*xy[1]**2+d*xy[1]+e
    x_income,y_price,z_birth=extract_array(extract_income_birth_price,column_order,[attr_income,attr_pr,attr_br])
    p0=0.00002,-0.0003,0.000003,-0.00005,0.3 #initial guess for a,b,c,d,e
    p1=curve_fit(func,(x_income,y_price),z_birth,p0) # fit values                   
    X_income,Y_price,Z_birth=get_2D_array(x_income,y_price,func,p1)
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    ax.plot_surface(X_income,Y_price,Z_birth)
    ax.set_xlabel('Income per Capita ($)')
    ax.set_ylabel('\n Median Home Price ($)')
    ax.set_zlabel('Birth Rate (%)')
    ax.set_title('Surface Plot of Birth Rate vs. Income&Home Price')
    plt.show()
    
# define global variables
database='..//data//Demography.db'
table_name_yr='Year'
table_name_st='State_abbreviation'
table_name_ct='County'
attr_ct,attr_br,attr_yr,attr_id,attr_pr='county','birth_rate','year','id','home_price'
db=sql.connect(database)
cur=db.cursor()
if __name__=='__main__':
    main()
