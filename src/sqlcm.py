def insert_statement(table,items,row,idx):
    # returns insertion statement for database table
    # table = table name
    # items = {column1:[],column2:[]}
    # row = the nth row
    # idx = id
    # return an execution statement for insertion
    result=[]
    split=', '
    result.append(f'INSERT INTO {table} ')
    columns=[]
    for i in items:
        columns.append(i)
    columns=split.join(columns)
    columns='id, '+columns
    result[0]+='('+columns+')'+' VALUES '
    values=[]
    for i in range(len(items)+1):
        values.append('?')
    values=split.join(values)
    result[0]+='('+values+')'
    values=[]
    for i in items:
        
        values.append(items[i][row])
    values.insert(0,idx)
    values=tuple(values)
    result.append(values)
    return tuple(result)
def create_statement(attribute,column_type,table_name):
    # returns creation statement for database table
    # attribute=a list of attribute names
    # column_type= a list of types of attributes
    # table_name=name of database table
    pair=[]
    split=', '
    for i,j in zip(attribute,column_type):
        pair.append(i+' '+j)
    return f'CREATE TABLE {table_name} ({split.join(pair)})'

def select_statement(table,columns,*values,select=[]):
    # returns selection statement for database table
    # table=table name
    # columns=[col1,col2] as condition
    # values=(val1,val2)
    # select=a list of the columns to be selected, [] means select all
    # returns the select statement
    value=[]
    
    if select:
        split=', '
        selection=split.join(select)
    else:
        
        selection='*'
    if values:
        split=' AND '
        condition=[]
        for i in columns:
            condition.append(i+'=?')
        condition=split.join(condition)
        return (f'SELECT {selection} FROM {table} WHERE {condition}', values)
    else:
        return (f'SELECT {selection} FROM {table}',)
