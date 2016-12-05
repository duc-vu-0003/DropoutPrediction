import random
import os
import pickle
import pandas as pd
import numpy as np
import os
import logging
from utils import Paths
import time
import collections
import encodings
import re
from os import path
# import MySQLdb
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from preprocess import *

def mergeResult():
    fu_raw = Paths.fu_file
    lack_of_birth = Paths.lack_of_birth
    lack_of_english_grade = Paths.lack_of_english_grade
    lack_of_village = Paths.lack_of_village

    dataFU = []
    dataBirth = []
    dataEnglish = []
    dataVillage = []

    if os.path.exists(fu_raw):
        dataFU = pd.read_csv(fu_raw, sep=",", encoding='utf-8', low_memory=False)

    if os.path.exists(lack_of_birth):
        dataBirth = pd.read_csv(lack_of_birth, sep=",", encoding='utf-8', low_memory=False)

    if os.path.exists(lack_of_english_grade):
        dataEnglish = pd.read_csv(lack_of_english_grade, sep=",", encoding='utf-8', low_memory=False)

    if os.path.exists(lack_of_village):
        dataVillage = pd.read_csv(lack_of_village, sep=",", encoding='utf-8', low_memory=False)

    # We need merge smaller file to bigger file
    for num in range(0, len(dataBirth)):
        rollNumber = dataBirth["RollNumber"][num]
        dob = dataBirth["DOB - Year (group)"][num]

        for num1 in range(0, len(dataFU)):
            if dataFU["RollNumber"][num1] == rollNumber:
                dataFU["DOB - Year (group)"][num1] = dob

    for num in range(0, len(dataEnglish)):
        rollNumber = dataEnglish["RollNumber"][num]
        dob = dataEnglish["EnglishAverageGrade"][num]

        for num1 in range(0, len(dataFU)):
            if dataFU["RollNumber"][num1] == rollNumber:
                dataFU["EnglishAverageGrade"][num1] = dob

    # print("Connecting to MySql...")
    # db = MySQLdb.connect("localhost", "root", "ngango", "FUWebLog", use_unicode=True, charset="utf8")
    # print("Connected to MySql - DB: FUWebLog")
    # cursor = db.cursor()
    #
    # for num in range(0, len(dataVillage)):
    #     dob = dataVillage["DistanceToHanoi"][num]
    #     print(dob)
    #     dataVillage["DistanceToHanoi"][num] = getDisanceFromVillage(db, cursor, dob)
    #     print(dataVillage["DistanceToHanoi"][num])
    #
    # # Rewrite data1 to file
    # db.close()
    # dataVillage.to_csv(Paths.eng_result_file, index=False, sep=",", encoding='UTF-8')

    for num in range(0, len(dataVillage)):
        rollNumber = dataVillage["RollNumber"][num]
        dob = dataVillage["DistanceToHanoi"][num]

        for num1 in range(0, len(dataFU)):
            if dataFU["RollNumber"][num1] == rollNumber:
                dataFU["DistanceToHanoi"][num1] = dob

    dataFU.to_csv(Paths.fu_result_file, index=False, sep=",", encoding='UTF-8')

def getDisanceFromVillage(db, cursor, village):
    sql = "SELECT driving_distance \
            FROM FUWebLog.distance_to_hanoi \
            WHERE city = '%s'" % village

    print(sql)

    try:
        # Execute the SQL command
        cursor.execute(sql)
        # Fetch all the rows in a list of lists.
        results = cursor.fetchall()
        return results[0][0]
    except Exception, e:
        print(e)
        return village

def isHaveTable(db, cursor):
    result = False
    sql = "SELECT * \
            FROM information_schema.tables \
            WHERE table_schema = 'FUWebLog' \
            AND table_name = 'FillMissingData' \
            LIMIT 1;"
    try:
        # Execute the SQL command
        cursor.execute(sql)
        # Fetch all the rows in a list of lists.
        results = cursor.fetchall()
        result = len(results) > 0
    except:
        result = False
    return result

def fillData():
    print("Connecting to MySql...")
    db = MySQLdb.connect("localhost", "root", "ngango", "FUWebLog", use_unicode=True, charset="utf8")
    print("Connected to MySql - DB: FUWebLog")
    cursor = db.cursor()

    creadTableIfNeed(db, cursor)

    dataVillage = []
    if os.path.exists(Paths.fu_result_file):
        dataVillage = pd.read_csv(Paths.fu_result_file, sep=",", encoding='utf-8', low_memory=False)
        dataVillage = dataVillage.where((pd.notnull(dataVillage)), None)
        print(dataVillage)
        importWebLog(dataVillage, db, cursor)

def creadTableIfNeed(db, cursor):
    print("FUWebLog: Check table FillMissingData...")
    if isHaveTable(db, cursor):
        print("FUWebLog: Already have table FillMissingData...")
        # return

    # Drop table if it already exist using execute() method.
    cursor.execute("DROP TABLE IF EXISTS FillMissingData")

    # Create table as per requirement
    sql = """CREATE TABLE FillMissingData (
             RollNumber CHAR(38) NOT NULL,
             DOB VARCHAR(255),
             EntryGrade VARCHAR(255),
             Major VARCHAR(255),
             DistanceToHanoi VARCHAR(255),
             EnglishAverageGrade VARCHAR(255))"""

    cursor.execute(sql)
    print("FUWebLog: Create table FillMissingData succesfully!!!")

def escapeString(value):
    return value.replace("nan", "")

def importWebLog(df, db, cursor):
    for index, row in df.iterrows():
        # Prepare SQL query to INSERT a record into the database.
        sql = "INSERT INTO FillMissingData(RollNumber, \
                                        DOB, \
                                        EntryGrade, \
                                        Major, \
                                        DistanceToHanoi, \
                                        EnglishAverageGrade) \
                VALUES (%s, %s, %s, %s , %s, %s)"

        try:
           # Execute the SQL command
           cursor.execute(sql, (row[0], row[1], row[2], row[3], row[4], row[5]))
           # Commit your changes in the database
           db.commit()
        except Exception as error:
           # Rollback in case there is any error
           print(error)
           db.rollback()

def main():
    oper = -1
    while int(oper) != 0:
        print('**************************************')
        print('Choose one of the following: ')
        print('1 - Pre Processing Raw Data')
        print('2 - Run with normal data')
        print('3 - Run with cloned data')
        print('4 - Run with cluster data')
        print('5 - Run All')
        print('0 - Exit')
        print('**************************************')
        oper = int(input("Enter your options: "))

        if oper == 0:
            exit()
        elif oper == 1:
            mergeResult()
        elif oper == 2:
            run(0)
        elif oper == 3:
            run(1)
        elif oper == 4:
            run(2)
        elif oper == 5:
            run(3)

if __name__ == "__main__":
    main()
