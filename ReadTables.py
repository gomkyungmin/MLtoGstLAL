#!/usr/bin/python

import argparse
import sqlite3
import numpy as np

def parse_command_line():

    usage = """Usage
    """
    parser = argparse.ArgumentParser(description=usage)

    parser.add_argument("--sqlite-file",action="store",type=str)
    parser.add_argument("--table-name",action="store",type=str)
    parser.add_argument("--column-name",action="store",type=str)

    args = vars(parser.parse_args())
    
    return args

    
def get_table_col_names(sqlfile,tablename,colname):

    conn = sqlite3.connect(sqlfile)
    conn.row_factory = lambda cursor, row: row[0]
    cursor = conn.cursor()

    cursor.execute("SELECT name from sqlite_master")
    tables = cursor.fetchall()
    # print "\nTables of %s:" % sqlfile
    # print tables    
    cursor.execute("SELECT * from {tn}".format(tn=tablename))
    columns = list(map(lambda x: x[0], cursor.description))
    print "\nColumns of %s:" % tablename
    print columns

    cursor.execute("SELECT {cn} from {tn}".format(cn=colname,tn=tablename))
    output = cursor.fetchall()
    output = np.array(output).reshape(len(output),1)
    print output
    
    
def main():

    args = parse_command_line()
    sqlfile = args['sqlite_file']
    tablename = args['table_name']
    colname = args['column_name']
    
    get_table_col_names(sqlfile,tablename,colname)
    
    
if __name__=='__main__':

    main()
