# Model 1 - Scalable Random Forest

import re
import numpy as np
import pandas as pd
import sqlite3
from pandas.io import sql
from datetime import datetime


# global variable to indicate the folder for all input / output files
FOLDER = './data/'


# Convert [lon,lat] string to list
def lonlat_convert(lonlat):

    lon = float(re.compile("[-+]?\d+.\d+").findall(lonlat)[0])
    lat = float(re.compile("[-+]?\d+.\d+").findall(lonlat)[1])
    combined = list()
    combined.append(lon)
    combined.append(lat)

    return combined


# Get Haversine distance
def get_dist(lonlat1, lonlat2):

    lon_diff = np.abs(lonlat1[0]-lonlat2[0])*np.pi/360.0
    lat_diff = np.abs(lonlat1[1]-lonlat2[1])*np.pi/360.0
    a = np.sin(lat_diff)**2 + np.cos(lonlat1[1]*np.pi/180.0) * np.cos(lonlat2[1]*np.pi/180.0) * np.sin(lon_diff)**2
    d = 2*6371*np.arctan2(np.sqrt(a), np.sqrt(1-a))

    return d


class CSVToSQL:

    def __init__(self, folder, file_in, file_out):

        self.folder = folder
        self.file_in = file_in
        self.file_out = file_out

    def generate_sqlite(self):

        print "Converting csv file to sqlite for train set:"
        num_lines = sum(1 for line in open(self.folder+self.file_in))
        columns = ['TRIP_ID', 'CALL_TYPE', 'ORIGIN_CALL', 'ORIGIN_STAND', 'TAXI_ID',
                   'TIMESTAMP', 'DAYTYPE', 'MISSING_DATA', 'POLYLINE']

        con = sqlite3.connect(self.folder+self.file_out)
        chunk_size = 5000
        count = 1

        for i in range(0, num_lines, chunk_size):

            df = pd.read_csv(self.folder+self.file_in, header=None,
                             nrows=chunk_size, skiprows=i, low_memory=False)
            df.columns = columns
            sql.to_sql(df, name='train_data', con=con, index=False,
                       index_label='molecule_id', if_exists='append')

            print "Batch No. {} completed".format(count)
            count += 1

        con.close()

        # Delete the first row with duplicate column names
        con = sqlite3.connect(self.folder+self.file_out)
        c = con.cursor()
        c.execute("DELETE FROM train_data WHERE TRIP_ID='TRIP_ID'")
        con.commit()
        con.close()

        print "All completed!\n"


class TrainDescriptive:

    def __init__(self, folder, file_in, file_out):

        self.folder = folder
        self.file_in = file_in
        self.file_out = file_out

    def transform(self):

        # this function does the following
        # 1) convert timestamp to time of the day, day of week, and month of year
        # 2) convert polyline to snapshots
        # 3) calculate the trip length
        # 4) calculate the average speed

        print "Generating training file with descriptive stats:"
        # initialize the connection with the input and output sqlite file
        con_in = sqlite3.connect(self.folder+self.file_in)
        con_out = sqlite3.connect(self.folder+self.file_out)

        con_in.text_factory = str
        chunk_reader = pd.read_sql("SELECT * FROM train_data WHERE MISSING_DATA!='1'",
                                   con_in, chunksize=5000)
        count = 1

        for chunk in chunk_reader:

            print 'Chunk {} started:'.format(count)
            chunk['Time_of_Day'] = chunk.TIMESTAMP.map(lambda x: datetime.utcfromtimestamp(float(x)).hour +
                                                                 datetime.utcfromtimestamp(float(x)).minute/60.0 +
                                                                 datetime.utcfromtimestamp(float(x)).second/3600.0)
            chunk['Hour_of_Day'] = chunk.Time_of_Day.map(lambda x: np.round(x))
            chunk['Day_of_Week'] = chunk.TIMESTAMP.map(lambda x: datetime.utcfromtimestamp(float(x)).weekday())
            chunk['Month_of_Year'] = chunk.TIMESTAMP.map(lambda x: datetime.utcfromtimestamp(float(x)).month)

            chunk['POLYLINE_Split'] = chunk.POLYLINE.map(lambda x:
                                                         re.compile("\[[-+]?\d+.\d+,[-+]?\d+.\d+\]").findall(x))
            chunk['Snapshots'] = chunk.POLYLINE_Split.map(lambda x: len(x))

            chunk = chunk[chunk.Snapshots > 10]
            chunk['Start_Point'] = chunk.POLYLINE_Split.map(lambda x: lonlat_convert(x[0]))
            chunk['End_Point'] = chunk.POLYLINE_Split.map(lambda x: lonlat_convert(x[-1]))

            chunk['Trip_Length'] = pd.DataFrame([get_dist(chunk.iloc[i].Start_Point, chunk.iloc[i].End_Point)
                                                 for i in range(len(chunk))])

            chunk['Avg_Speed'] = chunk['Trip_Length']*1000.0 / ((chunk['Snapshots']-1)*15)

            chunk.drop(['POLYLINE', 'POLYLINE_Split', 'Start_Point', 'End_Point'], axis=1, inplace=True)

            sql.to_sql(chunk,
                       name='train_data',
                       con=con_out,
                       index=False,
                       index_label='molecule_id',
                       if_exists='append')

            print 'Chunk {} completed!'.format(count)
            count += 1

        con_in.close()
        con_out.close()

        print "All completed!\n"

    def descriptive_hour(self):

        # generate descriptive file for each hour of the day
        print "Generating descriptive stats for each hour:"
        con = sqlite3.connect(self.folder+self.file_out)
        df_hourly = pd.read_sql("""
                                SELECT CAST(Hour_of_Day AS INTEGER) AS Hour_of_Day,
                                       (AVG(Snapshots)-1)*15 AS avg_trip_time,
                                        AVG(Trip_Length) AS avg_trip_length,
                                        COUNT(*) AS trip_count,
                                        AVG(Avg_Speed) AS avg_speed_per_trip
                                    FROM train_data
                                    GROUP BY Hour_of_Day
                                """, con)

        df_hourly.to_csv(self.folder+'Descriptive_Hour.csv', index=False)
        con.close()

        print "Completed!\n"

    def descriptive_weekday(self):

        # generate descriptive file for each weekday
        print "Generating descriptive stats for each weekday:"
        con = sqlite3.connect(self.folder+self.file_out)
        df_weekday = pd.read_sql("""
                                 SELECT Day_of_Week,
                                        (AVG(Snapshots)-1)*15 AS avg_trip_time,
                                         AVG(Trip_Length) AS avg_trip_length,
                                         COUNT(*) AS trip_count,
                                         AVG(Avg_Speed) AS avg_speed_per_trip
                                    FROM train_data
                                    GROUP BY Day_of_Week
                                 """, con)

        df_weekday.to_csv(self.folder+'Descriptive_Weekday.csv', index=False)
        con.close()

        print "Completed!\n"

    def descriptive_month(self):
        # generate descriptive file for each month
        print "Generating descriptive stats for each month:"
        con = sqlite3.connect(self.folder+self.file_out)
        df_month = pd.read_sql("""
                               SELECT Month_of_Year,
                                      (AVG(Snapshots)-1)*15 AS avg_trip_time,
                                       AVG(Trip_Length) AS avg_trip_length,
                                       COUNT(*) AS trip_count,
                                       AVG(Avg_Speed) AS avg_speed_per_trip
                                    FROM train_data
                                    GROUP BY Month_of_Year
                               """, con)
        df_month.to_csv(self.folder+'Descriptive_Month.csv', index=False)
        con.close()

        print "Completed!\n"

    def descriptive_driver(self):

        # generate descriptive file for each driver id
        print "Generating descriptive stats for each driver:"
        con = sqlite3.connect(self.folder+self.file_out)
        df_driver = pd.read_sql("""
                                SELECT TAXI_ID, (avg(Snapshots)-1)*15 AS avg_trip_time,
                                       AVG(Trip_Length) AS avg_trip_length,
                                       COUNT(*) AS trip_count,
                                       AVG(Avg_Speed) AS avg_speed_per_trip
                                    FROM train_data
                                    GROUP BY TAXI_ID
                                """, con)

        df_driver.to_csv(self.folder+'Descriptive_Driver.csv', index=False)
        con.close()

        print "Completed!\n"

    def descriptive_stand(self):

        # generate descriptive file for each taxi stand
        print "Generating descriptive stats for each taxi stand:"
        con = sqlite3.connect(self.folder+self.file_out)
        df_stand = pd.read_sql("""
                               SELECT CAST(ORIGIN_STAND AS INTEGER) AS ORIGIN_STAND,
                                      (AVG(Snapshots)-1)*15 AS avg_trip_time,
                                       AVG(Trip_Length) AS avg_trip_length,
                                       COUNT(*) AS trip_count,
                                       AVG(Avg_Speed) AS avg_speed_per_trip
                                    FROM train_data
                                    WHERE CALL_TYPE='B'
                                    GROUP BY CAST(ORIGIN_STAND AS INTEGER)
                                """, con)

        df_stand.to_csv(self.folder+'Descriptive_Stand.csv', index=False)
        con.close()

        print "Completed!\n"

    def descriptive_caller(self):

        # generate descriptive file for each caller id
        print "Generating descriptive stats for each caller:"
        con = sqlite3.connect(self.folder+self.file_out)
        df_caller = pd.read_sql("""
                                SELECT CAST(ORIGIN_CALL AS INTEGER) AS ORIGIN_CALL,
                                       (AVG(Snapshots)-1)*15 AS avg_trip_time,
                                        AVG(Trip_Length) AS avg_trip_length,
                                        COUNT(*) AS trip_count,
                                        AVG(Avg_Speed) AS avg_speed_per_trip
                                    FROM train_data
                                    WHERE CALL_TYPE='A'
                                    GROUP BY CAST(ORIGIN_CALL AS INTEGER)
                                """, con)

        df_caller.to_csv(self.folder+'Descriptive_Caller.csv', index=False)
        con.close()

        print "Completed!\n"


class TrainPreProcessing:

    def __init__(self, folder, file_in, file_out):

        # initialize folder and input / output file names
        self.folder = folder
        self.file_in = file_in
        self.file_out = file_out

    def train_pre_process(self):

        # read in the training set by chunks, and add engineered features
        print "Pre-processing the training set:"
        chunk_reader = pd.read_csv(self.folder+self.file_in, chunksize=5000)
        count = 1

        for chunk in chunk_reader:

            print "Chunk No.{} started:".format(count)

            # reset index
            chunk = chunk[chunk.MISSING_DATA == False]
            chunk.reset_index(inplace=True)

            # split the polyline and calculate actual snapshots and travel time
            chunk['POLYLINE_Split'] = chunk.POLYLINE.map(lambda x:
                                                         re.compile("\[[-+]?\d+.\d+,[-+]?\d+.\d+\]").findall(x))
            chunk['Snapshots'] = chunk.POLYLINE_Split.map(lambda x: len(x))
            chunk = pd.DataFrame(chunk[chunk.Snapshots > 10])
            chunk['Travel_Time'] = chunk['Snapshots'].map(lambda x: (x-1)*15)

            # Randomly truncate to match the format of the test data
            def truncate_func(row):

                path_len = np.random.randint(1, row['Snapshots'])
                return tuple(row['POLYLINE_Split'][:path_len])

            chunk['POLYLINE_Split_Truncated'] = chunk.apply(truncate_func, axis=1)

            # Delete/rename columns
            chunk.drop(['POLYLINE', 'POLYLINE_Split'], axis=1, inplace=True)
            chunk.rename(columns={'POLYLINE_Split_Truncated': 'POLYLINE_Split'}, inplace=True)

            # Add dummies for CALL_TYPE
            chunk = pd.concat([chunk, pd.get_dummies(chunk.CALL_TYPE, prefix='Call_Type_')], axis=1)

            # Deal with time stamp
            chunk['Time_of_Day'] = chunk.TIMESTAMP.map(lambda x:
                                                       datetime.utcfromtimestamp(float(x)).hour +
                                                       datetime.utcfromtimestamp(float(x)).minute/60.0 +
                                                       datetime.utcfromtimestamp(float(x)).second/3600.0)
            chunk['Hour_of_Day'] = chunk.Time_of_Day.map(lambda x: np.round(x)).astype(int)
            chunk['Day_of_Week'] = chunk.TIMESTAMP.map(lambda x: datetime.utcfromtimestamp(float(x)).weekday())
            chunk['Month_of_Year'] = chunk.TIMESTAMP.map(lambda x: datetime.utcfromtimestamp(float(x)).month)

            # Read in description for hour of the day
            file1 = 'Descriptive_Hour.csv'
            df_hour = pd.read_csv(self.folder+file1, header=False)
            # Join by hour of the day
            chunk = pd.merge(chunk, df_hour, on='Hour_of_Day')
            chunk['Hour_TT'] = chunk.avg_trip_time
            chunk['Hour_TL'] = chunk.avg_trip_length
            chunk['Hour_TC'] = chunk.trip_count
            chunk['Hour_TS'] = chunk.avg_speed_per_trip
            chunk.drop(['avg_trip_time', 'avg_trip_length', 'trip_count', 'avg_speed_per_trip'], axis=1, inplace=True)

            # Read in description for day of the week
            file2 = 'Descriptive_Weekday.csv'
            df_weekday = pd.read_csv(self.folder+file2, header=False)
            # Join by day of the week
            chunk = pd.merge(chunk, df_weekday, on='Day_of_Week')
            chunk['Weekday_TT'] = chunk.avg_trip_time
            chunk['Weekday_TL'] = chunk.avg_trip_length
            chunk['Weekday_TC'] = chunk.trip_count
            chunk['Weekday_TS'] = chunk.avg_speed_per_trip
            chunk.drop(['avg_trip_time', 'avg_trip_length', 'trip_count', 'avg_speed_per_trip'], axis=1, inplace=True)

            # Read in description for month of the year
            file3 = 'Descriptive_Month.csv'
            df_month = pd.read_csv(self.folder+file3, header=False)
            # Join by month of the year
            chunk = pd.merge(chunk, df_month, on='Month_of_Year')
            chunk['Month_TT'] = chunk.avg_trip_time
            chunk['Month_TL'] = chunk.avg_trip_length
            chunk['Month_TC'] = chunk.trip_count
            chunk['Month_TS'] = chunk.avg_speed_per_trip
            chunk.drop(['avg_trip_time', 'avg_trip_length', 'trip_count', 'avg_speed_per_trip'], axis=1, inplace=True)

            # Read in description for driver id
            file4 = 'Descriptive_Driver.csv'
            df_driver = pd.read_csv(self.folder+file4, header=False)
            # Join by driver id
            chunk = pd.merge(chunk, df_driver, on='TAXI_ID')
            chunk['Driver_TT'] = chunk.avg_trip_time
            chunk['Driver_TL'] = chunk.avg_trip_length
            chunk['Driver_TC'] = chunk.trip_count
            chunk['Driver_TS'] = chunk.avg_speed_per_trip
            chunk.drop(['avg_trip_time', 'avg_trip_length', 'trip_count', 'avg_speed_per_trip'], axis=1, inplace=True)

            # Read in description for stand id
            file5 = 'Descriptive_Stand.csv'
            df_stand = pd.read_csv(self.folder+file5, header=False)
            # Left Join by stand id
            chunk = pd.merge(chunk, df_stand, how='left', on=['ORIGIN_STAND', 'ORIGIN_STAND'])
            chunk['Stand_TT'] = chunk.avg_trip_time
            chunk['Stand_TL'] = chunk.avg_trip_length
            chunk['Stand_TC'] = chunk.trip_count
            chunk['Stand_TS'] = chunk.avg_speed_per_trip
            chunk.drop(['avg_trip_time', 'avg_trip_length', 'trip_count', 'avg_speed_per_trip'], axis=1, inplace=True)

            # Read in description for caller id
            file6 = 'Descriptive_Caller.csv'
            df_caller = pd.read_csv(self.folder+file6, header=False)
            # Left Join by caller id
            chunk = pd.merge(chunk, df_caller, how='left', on=['ORIGIN_CALL', 'ORIGIN_CALL'])
            chunk['Caller_TT'] = chunk.avg_trip_time
            chunk['Caller_TL'] = chunk.avg_trip_length
            chunk['Caller_TC'] = chunk.trip_count
            chunk['Caller_TS'] = chunk.avg_speed_per_trip
            chunk.drop(['avg_trip_time', 'avg_trip_length', 'trip_count', 'avg_speed_per_trip'], axis=1, inplace=True)

            # If stand id is null, we assign grand average to the stand description
            chunk.loc[chunk.ORIGIN_STAND.isnull(), 'Stand_TT'] = 671.847205828125
            chunk.loc[chunk.ORIGIN_STAND.isnull(), 'Stand_TL'] = 3.41625640673437
            chunk.loc[chunk.ORIGIN_STAND.isnull(), 'Stand_TC'] = 12459.53125
            chunk.loc[chunk.ORIGIN_STAND.isnull(), 'Stand_TS'] = 6.77996522545313

            # If caller id is null, we assign average numbers to the caller description
            chunk.loc[chunk.ORIGIN_CALL.isnull(), 'Caller_TT'] = 769.644426032955
            chunk.loc[chunk.ORIGIN_CALL.isnull(), 'Caller_TL'] = 3.45908442749228
            chunk.loc[chunk.ORIGIN_CALL.isnull(), 'Caller_TC'] = 6.33404623868778
            chunk.loc[chunk.ORIGIN_CALL.isnull(), 'Caller_TS'] = 5.92595987288811

            # If there are still null values for stand descriptions
            chunk.loc[chunk.Stand_TT.isnull(), 'Stand_TT'] = 671.847205828125
            chunk.loc[chunk.Stand_TL.isnull(), 'Stand_TL'] = 3.41625640673437
            chunk.loc[chunk.Stand_TC.isnull(), 'Stand_TC'] = 12459.53125
            chunk.loc[chunk.Stand_TS.isnull(), 'Stand_TS'] = 6.77996522545313

            # If there are still null values for caller descriptions
            chunk.loc[chunk.Caller_TT.isnull(), 'Caller_TT'] = 769.644426032955
            chunk.loc[chunk.Caller_TL.isnull(), 'Caller_TL'] = 3.45908442749228
            chunk.loc[chunk.Caller_TC.isnull(), 'Caller_TC'] = 6.33404623868778
            chunk.loc[chunk.Caller_TS.isnull(), 'Caller_TS'] = 5.92595987288811

            # Add start speed (if less than 2 snapshots, use average start speed)
            def get_start_speed(POLYLINE_Split):

                num = len(POLYLINE_Split)

                if num < 2:
                    return None

                else:
                    Lonlat_first = lonlat_convert(POLYLINE_Split[0])
                    Lonlat_second = lonlat_convert(POLYLINE_Split[1])
                    start_speed = get_dist(Lonlat_first, Lonlat_second) * 1000.0 / 15.0

                    return start_speed

            chunk['Start_Speed'] = chunk.POLYLINE_Split.map(lambda x: get_start_speed(x))

            # Add end speed (if less than 2 snapshots, use average end speed)
            def get_end_speed(POLYLINE_Split):

                num = len(POLYLINE_Split)

                if num < 2:
                    return None

                else:
                    Lonlat_last_but_one = lonlat_convert(POLYLINE_Split[num-2])
                    Lonlat_last = lonlat_convert(POLYLINE_Split[num-1])
                    end_speed = get_dist(Lonlat_last_but_one, Lonlat_last) * 1000.0 / 15.0

                    return end_speed

            chunk['End_Speed'] = chunk.POLYLINE_Split.map(lambda x: get_end_speed(x))

            # Add average speed (if less than 2 snapshots, use average average speed
            def get_avg_speed(POLYLINE_Split):

                num = len(POLYLINE_Split)

                if num < 2:
                    return None

                else:
                    speeds = []
                    for i in range(num-1):
                        Lonlat_one = lonlat_convert(POLYLINE_Split[i])
                        Lonlat_two = lonlat_convert(POLYLINE_Split[i+1])
                        speed = get_dist(Lonlat_one, Lonlat_two) * 1000.0 / 15.0
                        speeds.append(speed)

                    return np.mean(speeds)

            chunk['Avg_Speed'] = chunk.POLYLINE_Split.map(lambda x: get_avg_speed(x))

            # Add start speed two
            def get_start_speed_two(POLYLINE_Split):

                num = len(POLYLINE_Split)

                if num < 3:
                    return None

                else:
                    Lonlat_second = lonlat_convert(POLYLINE_Split[1])
                    Lonlat_third = lonlat_convert(POLYLINE_Split[2])
                    start_speed_two = get_dist(Lonlat_second, Lonlat_third) * 1000.0 / 15.0

                    return start_speed_two

            chunk['Start_Speed_two'] = chunk.POLYLINE_Split.map(lambda x: get_start_speed_two(x))

            # Add end speed two
            def get_end_speed_two(POLYLINE_Split):

                num = len(POLYLINE_Split)

                if num < 3:
                    return None

                else:
                    Lonlat_last_but_two = lonlat_convert(POLYLINE_Split[num-3])
                    Lonlat_last_but_one = lonlat_convert(POLYLINE_Split[num-2])
                    end_speed_two = get_dist(Lonlat_last_but_two, Lonlat_last_but_one) * 1000.0 / 15.0

                    return end_speed_two

            chunk['End_Speed_two'] = chunk.POLYLINE_Split.map(lambda x: get_end_speed_two(x))

            # Add current snapshots
            chunk['Current_Snapshots'] = chunk.POLYLINE_Split.map(lambda x: len(x))
            chunk['Current_Snapshots_log'] = chunk.POLYLINE_Split.map(lambda x: np.log(len(x)+1))

            # This is for generating the cleaned training set
            chunk_out = chunk[['TRIP_ID', 'CALL_TYPE', 'ORIGIN_CALL', 'ORIGIN_STAND', 'TAXI_ID', 'TIMESTAMP',
                               'Call_Type__A', 'Call_Type__B', 'Call_Type__C',
                               'Time_of_Day', 'Hour_of_Day', 'Day_of_Week', 'Month_of_Year',
                               'Hour_TT', 'Hour_TL', 'Hour_TC', 'Hour_TS',
                               'Weekday_TT', 'Weekday_TL', 'Weekday_TC', 'Weekday_TS',
                               'Month_TT', 'Month_TL', 'Month_TC', 'Month_TS',
                               'Driver_TT', 'Driver_TL', 'Driver_TC', 'Driver_TS',
                               'Stand_TT', 'Stand_TL', 'Stand_TC', 'Stand_TS',
                               'Caller_TT', 'Caller_TL', 'Caller_TC', 'Caller_TS',
                               'Start_Speed', 'End_Speed', 'Avg_Speed', 'Start_Speed_two', 'End_Speed_two',
                               'Current_Snapshots', 'Current_Snapshots_log',
                               'Snapshots', 'Travel_Time', 'POLYLINE_Split']]
            chunk = []

            if count == 1:
                chunk_out.to_csv(self.folder+'train_cleaned_temp.csv', mode='a', index=False)
            else:
                chunk_out.to_csv(self.folder+'train_cleaned_temp.csv', mode='a', header=False, index=False)

            print 'Chunk No.{} completed!'.format(count)
            count += 1

        print "All completed!\n"

    def fix_null(self):

        # fix the null values of speed variables using grand average
        print "Fixing null values in speed variables:"
        speed_dict = {'Start_Speed': 2.255119,
                      'End_Speed': 7.652231,
                      'Avg_Speed': 6.905948,
                      'Start_Speed_Two': 4.302278,
                      'End_Speed_Two': 7.619596}

        chunk_reader = pd.read_csv(self.folder+'train_cleaned_temp.csv', chunksize=10000)
        count = 1

        for chunk in chunk_reader:

            print 'Chunk No.{} started:'.format(count)

            chunk = chunk[(chunk.Start_Speed <= 40) & (chunk.End_Speed <= 40) & (chunk.Avg_Speed <= 40) &
                    (chunk.Start_Speed_two <= 40) & (chunk.End_Speed_two <= 40) & (chunk.Current_Snapshots < 1000)]

            chunk.reset_index(inplace=True)
            chunk.drop(['index'], axis=1, inplace=True)

            chunk.loc[chunk.Start_Speed.isnull(), 'Start_Speed'] = speed_dict['Start_Speed']
            chunk.loc[chunk.End_Speed.isnull(), 'End_Speed'] = speed_dict['End_Speed']
            chunk.loc[chunk.Avg_Speed.isnull(), 'Avg_Speed'] = speed_dict['Avg_Speed']
            chunk.loc[chunk.Start_Speed_two.isnull(), 'Start_Speed_two'] = speed_dict['Start_Speed_Two']
            chunk.loc[chunk.End_Speed_two.isnull(), 'End_Speed_two'] = speed_dict['End_Speed_Two']

            # Save the changes to a new training file
            if count == 1:
                chunk.to_csv(self.folder+self.file_out, mode='a', index=False)
            else:
                chunk.to_csv(self.folder+self.file_out, mode='a', header=False, index=False)

            print 'Chunk No.{} completed!'.format(count)
            count += 1

        print "All completed!\n"


class TestPreProcessing:

    def __init__(self, folder, file_in, file_out):

        self.folder = folder
        self.file_in = file_in
        self.file_out = file_out

    def test_pre_process(self):

        print "Pre-processing the training set:"

        # perform the same feature engineering as the training set
        df_Test = pd.read_csv(self.folder+self.file_in)

        # Need to keep track the the trip_id
        df_Test.reset_index(inplace=True)

        # Add dummies for CALL_TYPE
        df_Test = pd.concat([df_Test, pd.get_dummies(df_Test.CALL_TYPE, prefix='Call_Type_')], axis=1)

        # Deal with time stamp
        df_Test['Time_of_Day'] = df_Test.TIMESTAMP.map(lambda x: datetime.utcfromtimestamp(x).hour +
                                                       datetime.utcfromtimestamp(float(x)).minute/60.0 +
                                                       datetime.utcfromtimestamp(float(x)).second/3600.0)
        df_Test['Hour_of_Day'] = df_Test.Time_of_Day.map(lambda x: np.round(x)).astype(int)
        df_Test['Day_of_Week'] = df_Test.TIMESTAMP.map(lambda x: datetime.utcfromtimestamp(float(x)).weekday())
        df_Test['Month_of_Year'] = df_Test.TIMESTAMP.map(lambda x: datetime.utcfromtimestamp(float(x)).month)

        # Read in description for hour of the day
        file1 = 'Descriptive_Hour.csv'
        df_hour = pd.read_csv(self.folder+file1, header=False)
        # Join by hour of the day
        df_Test = pd.merge(df_Test, df_hour, on='Hour_of_Day')
        df_Test['Hour_TT'] = df_Test.avg_trip_time
        df_Test['Hour_TL'] = df_Test.avg_trip_length
        df_Test['Hour_TC'] = df_Test.trip_count
        df_Test['Hour_TS'] = df_Test.avg_speed_per_trip
        df_Test.drop(['avg_trip_time', 'avg_trip_length', 'trip_count', 'avg_speed_per_trip'], axis=1, inplace=True)

        # Read in description for day of the week
        file2 = 'Descriptive_Weekday.csv'
        df_weekday = pd.read_csv(self.folder+file2, header=False)
        # Join by day of week
        df_Test = pd.merge(df_Test, df_weekday, on='Day_of_Week')
        df_Test['Weekday_TT'] = df_Test.avg_trip_time
        df_Test['Weekday_TL'] = df_Test.avg_trip_length
        df_Test['Weekday_TC'] = df_Test.trip_count
        df_Test['Weekday_TS'] = df_Test.avg_speed_per_trip
        df_Test.drop(['avg_trip_time', 'avg_trip_length', 'trip_count', 'avg_speed_per_trip'], axis=1, inplace=True)

        # Read in description for month of the year
        file3 = 'Descriptive_Month.csv'
        df_month = pd.read_csv(self.folder+file3, header=False)
        # Join by month of year
        df_Test = pd.merge(df_Test, df_month, on='Month_of_Year')
        df_Test['Month_TT'] = df_Test.avg_trip_time
        df_Test['Month_TL'] = df_Test.avg_trip_length
        df_Test['Month_TC'] = df_Test.trip_count
        df_Test['Month_TS'] = df_Test.avg_speed_per_trip
        df_Test.drop(['avg_trip_time', 'avg_trip_length', 'trip_count', 'avg_speed_per_trip'], axis=1, inplace=True)

        # Read in description for driver id
        file4 = 'Descriptive_Driver.csv'
        df_driver = pd.read_csv(self.folder+file4, header=False)
        # Join by driver id
        df_Test = pd.merge(df_Test, df_driver, on='TAXI_ID')
        df_Test['Driver_TT'] = df_Test.avg_trip_time
        df_Test['Driver_TL'] = df_Test.avg_trip_length
        df_Test['Driver_TC'] = df_Test.trip_count
        df_Test['Driver_TS'] = df_Test.avg_speed_per_trip
        df_Test.drop(['avg_trip_time', 'avg_trip_length', 'trip_count', 'avg_speed_per_trip'], axis=1, inplace=True)

        # Read in description for stand id
        file5 = 'Descriptive_Stand.csv'
        df_stand = pd.read_csv(self.folder+file5, header=False)
        # Left Join by stand id
        df_Test = pd.merge(df_Test, df_stand, how='left', on=['ORIGIN_STAND', 'ORIGIN_STAND'])
        df_Test['Stand_TT'] = df_Test.avg_trip_time
        df_Test['Stand_TL'] = df_Test.avg_trip_length
        df_Test['Stand_TC'] = df_Test.trip_count
        df_Test['Stand_TS'] = df_Test.avg_speed_per_trip
        df_Test.drop(['avg_trip_time', 'avg_trip_length', 'trip_count', 'avg_speed_per_trip'], axis=1, inplace=True)

        # Read in description for caller id
        file6 = 'Descriptive_Caller.csv'
        df_caller = pd.read_csv(self.folder+file6, header=False)
        # Left Join by caller id
        df_Test = pd.merge(df_Test, df_caller, how='left', on=['ORIGIN_CALL', 'ORIGIN_CALL'])
        df_Test['Caller_TT'] = df_Test.avg_trip_time
        df_Test['Caller_TL'] = df_Test.avg_trip_length
        df_Test['Caller_TC'] = df_Test.trip_count
        df_Test['Caller_TS'] = df_Test.avg_speed_per_trip
        df_Test.drop(['avg_trip_time', 'avg_trip_length', 'trip_count', 'avg_speed_per_trip'], axis=1, inplace=True)

        # If stand id is null, we assign average numbers to the stand description
        df_Test.loc[df_Test.ORIGIN_STAND.isnull(), 'Stand_TT'] = 671.847205828125
        df_Test.loc[df_Test.ORIGIN_STAND.isnull(), 'Stand_TL'] = 3.41625640673437
        df_Test.loc[df_Test.ORIGIN_STAND.isnull(), 'Stand_TC'] = 12459.53125
        df_Test.loc[df_Test.ORIGIN_STAND.isnull(), 'Stand_TS'] = 6.77996522545313

        # If caller id is null, we assign average numbers to the caller description
        df_Test.loc[df_Test.ORIGIN_CALL.isnull(), 'Caller_TT'] = 769.644426032955
        df_Test.loc[df_Test.ORIGIN_CALL.isnull(), 'Caller_TL'] = 3.45908442749228
        df_Test.loc[df_Test.ORIGIN_CALL.isnull(), 'Caller_TC'] = 6.33404623868778
        df_Test.loc[df_Test.ORIGIN_CALL.isnull(), 'Caller_TS'] = 5.92595987288811

        # four special cases here
        df_Test.loc[df_Test.Caller_TL.isnull(), 'Caller_TL'] = 3.45908442749228
        df_Test.loc[df_Test.Caller_TS.isnull(), 'Caller_TS'] = 5.92595987288811
        df_Test.loc[df_Test.Caller_TT.isnull(), 'Caller_TT'] = 769.644426032955
        df_Test.loc[df_Test.Caller_TC.isnull(), 'Caller_TC'] = 6.33404623868778

        # Don't forget this step!
        df_Test['POLYLINE_Split'] = df_Test.POLYLINE.map(lambda x:
                                                         re.compile("\[[-+]?\d+.\d+,[-+]?\d+.\d+\]").findall(x))

        speed_dict = {'Start_Speed': 2.255119,
                      'End_Speed': 7.652231,
                      'Avg_Speed': 6.905948,
                      'Start_Speed_Two': 4.302278,
                      'End_Speed_Two': 7.619596}

        # Add start speed (if less than 2 snapshots, use average start speed)
        def get_start_speed(POLYLINE_Split):

            num = len(POLYLINE_Split)

            if num < 2:
                return None

            else:
                Lonlat_first = lonlat_convert(POLYLINE_Split[0])
                Lonlat_second = lonlat_convert(POLYLINE_Split[1])
                start_speed = get_dist(Lonlat_first, Lonlat_second) * 1000.0 / 15.0

                return start_speed

        df_Test['Start_Speed'] = df_Test.POLYLINE_Split.map(lambda x: get_start_speed(x))
        df_Test.loc[df_Test.Start_Speed.isnull(), 'Start_Speed'] = speed_dict['Start_Speed']

        # Add end speed (if less than 2 snapshots, use average end speed)
        def get_end_speed(POLYLINE_Split):

            num = len(POLYLINE_Split)

            if num < 2:
                return None

            else:
                Lonlat_last_but_one = lonlat_convert(POLYLINE_Split[num-2])
                Lonlat_last = lonlat_convert(POLYLINE_Split[num-1])
                end_speed = get_dist(Lonlat_last_but_one, Lonlat_last) * 1000.0 / 15.0

                return end_speed

        df_Test['End_Speed'] = df_Test.POLYLINE_Split.map(lambda x: get_end_speed(x))
        df_Test.loc[df_Test.End_Speed.isnull(), 'End_Speed'] = speed_dict['End_Speed']

        # Add average speed (if less than 2 snapshots, use average average speed
        def get_avg_speed(POLYLINE_Split):

            num = len(POLYLINE_Split)

            if num < 2:
                return None
            else:
                speeds = []
                for i in range(num-1):
                    Lonlat_one = lonlat_convert(POLYLINE_Split[i])
                    Lonlat_two = lonlat_convert(POLYLINE_Split[i+1])
                    speed = get_dist(Lonlat_one, Lonlat_two)*1000.0/15.0
                    speeds.append(speed)

                return np.mean(speeds)

        df_Test['Avg_Speed'] = df_Test.POLYLINE_Split.map(lambda x: get_avg_speed(x))
        df_Test.loc[df_Test.Avg_Speed.isnull(), 'Avg_Speed'] = speed_dict['Avg_Speed']

        # Add Start_Speed_two
        def get_start_speed_two(POLYLINE_Split):

            num = len(POLYLINE_Split)

            if num < 3:
                return None

            else:
                Lonlat_second = lonlat_convert(POLYLINE_Split[1])
                Lonlat_third = lonlat_convert(POLYLINE_Split[2])
                start_speed_two = get_dist(Lonlat_second, Lonlat_third) * 1000.0 / 15.0

                return start_speed_two

        df_Test['Start_Speed_two'] = df_Test.POLYLINE_Split.map(lambda x: get_start_speed_two(x))
        df_Test.loc[df_Test.Start_Speed_two.isnull(), 'Start_Speed_two'] = speed_dict['Start_Speed_Two']

        # Add End_Speed_two
        def get_end_speed_two(POLYLINE_Split):

            num = len(POLYLINE_Split)

            if num < 3:
                return None

            else:
                Lonlat_last_but_two = lonlat_convert(POLYLINE_Split[num-3])
                Lonlat_last_but_one = lonlat_convert(POLYLINE_Split[num-2])
                end_speed_two = get_dist(Lonlat_last_but_two, Lonlat_last_but_one)*1000.0/15.0

                return end_speed_two

        df_Test['End_Speed_two'] = df_Test.POLYLINE_Split.map(lambda x: get_end_speed_two(x))
        df_Test.loc[df_Test.End_Speed_two.isnull(), 'End_Speed_two'] = speed_dict['End_Speed_Two']

        # Add current snapshots
        df_Test['Current_Snapshots'] = df_Test.POLYLINE_Split.map(lambda x: len(x))
        df_Test['Current_Snapshots_log'] = df_Test.POLYLINE_Split.map(lambda x: np.log(len(x)+1))

        df_Test.sort(['index'], ascending=1, inplace=True)
        df_Test.reset_index(inplace=True)
        df_Test.drop(['level_0', 'index'], axis=1, inplace=True)

        # For test examples, prepare the X and y
        X_test_with_id = df_Test[['TRIP_ID', 'Call_Type__A', 'Call_Type__B', 'Call_Type__C',
                                  'Time_of_Day', 'Hour_of_Day', 'Day_of_Week', 'Month_of_Year',
                                  'Hour_TT', 'Hour_TL', 'Hour_TC', 'Hour_TS',
                                  'Weekday_TT', 'Weekday_TL', 'Weekday_TC', 'Weekday_TS',
                                  'Month_TT', 'Month_TL', 'Month_TC', 'Month_TS',
                                  'Driver_TT', 'Driver_TL', 'Driver_TC', 'Driver_TS',
                                  'Stand_TT', 'Stand_TL', 'Stand_TC', 'Stand_TS',
                                  'Caller_TT', 'Caller_TL', 'Caller_TC', 'Caller_TS',
                                  'Start_Speed', 'End_Speed', 'Avg_Speed', 'Start_Speed_two', 'End_Speed_two',
                                  'Current_Snapshots', 'Current_Snapshots_log']]

        X_test_with_id.to_csv(self.folder+self.file_out, index=False)

        print "Completed!\n"


class ScalableRandomForest:

    def __init__(self, num_trees, q, folder, train_file, test_file):

        # num_trees is used to determine number of trees in random forest
        # it is also the same value used to decide number of partitions for total data set
        self.num_trees = num_trees
        self.q = q
        self.folder = folder
        self.train_file = train_file
        self.test_file = test_file
        self.split_files = None

    def create_file_names(self):

        # Create file name for each split database
        self.split_files = []

        for i in range(self.num_trees):
            self.split_files.append('Train_Part{}.csv'.format(i))

    def split_train(self):

        # split the training set into same number of partitions as the number of trees
        print "Splitting the training set:"
        chunk_reader = pd.read_csv(self.folder+self.train_file, chunksize=50000)
        count = 1

        for chunk in chunk_reader:
            print 'Chunk No.{} started:'.format(count)
            chunk = chunk[(chunk.Start_Speed <= 40) & (chunk.End_Speed <= 40) & (chunk.Avg_Speed <= 40) &
                    (chunk.Start_Speed_two <= 40) & (chunk.End_Speed_two <= 40) & (chunk.Current_Snapshots < 1000)]
            # Add row number
            chunk.reset_index(inplace=True)
            chunk['ind'] = chunk.index.astype(int)
            chunk.drop(['index'], axis=1, inplace=True)

            for i in range(self.num_trees):

                sample = pd.DataFrame(chunk[chunk.ind % self.num_trees == i])
                sample.drop(['ind'], axis=1, inplace=True)
                # First chunk, keep the header
                if count == 1:
                    sample.to_csv(self.folder+self.split_files[i], index=False, mode='a')
                # Second and later chunk, don't keep the header
                else:
                    sample.to_csv(self.folder+self.split_files[i], index=False, mode='a', header=False)

            print 'Chunk No.{} completed!'.format(count)
            count += 1

        print "All completed!\n"

    def train_predict_by_partition(self):

        # for each split data set, train with a base tree
        from sklearn.ensemble import RandomForestRegressor

        print "Training Scalable Random Forest:"

        for i in range(self.num_trees):

            print "Partition {} started:".format(i)
            df = pd.read_csv(self.folder+self.split_files[i])

            df_X = df[['Call_Type__A', 'Call_Type__B', 'Call_Type__C',
                        'Time_of_Day', 'Hour_of_Day', 'Day_of_Week', 'Month_of_Year',
                        'Hour_TT', 'Hour_TL', 'Hour_TC', 'Hour_TS',
                        'Weekday_TT', 'Weekday_TL', 'Weekday_TC', 'Weekday_TS',
                        'Month_TT', 'Month_TL', 'Month_TC', 'Month_TS',
                        'Driver_TT', 'Driver_TL', 'Driver_TC', 'Driver_TS',
                        'Stand_TT', 'Stand_TL', 'Stand_TC', 'Stand_TS',
                        'Caller_TT', 'Caller_TL', 'Caller_TC', 'Caller_TS',
                        'Start_Speed', 'End_Speed', 'Avg_Speed', 'Start_Speed_two', 'End_Speed_two',
                        'Current_Snapshots', 'Current_Snapshots_log']]

            y = np.log(df['Travel_Time']).values

            # Randomly select subset of q*num_features attributes
            column_list = df_X.columns.tolist()
            num_features = len(column_list)
            ind_selected = np.random.permutation(num_features)[:int(num_features * self.q)]
            feature_selected = [column_list[k] for k in ind_selected]

            # Prepare X
            X = df_X[feature_selected].values

            # Prepare X_test
            df_test = pd.read_csv(self.folder+self.test_file)
            X_test = df_test[feature_selected].values

            # Prepare test id
            IDs = df_test['TRIP_ID'].values

            # Train the single tree
            maxFeatures = int(np.sqrt(num_features))
            clf = RandomForestRegressor(n_estimators=100, bootstrap=False, max_features=maxFeatures)
            clf = clf.fit(X, y)

            # Predict the test
            y_test = np.exp(clf.predict(X_test))

            # Save the predictions
            result = np.c_[IDs, y_test]
            df_result = pd.DataFrame(result, columns=['TRIP_ID', 'TRAVEL_TIME'])
            df_result.to_csv(self.folder+'submission_train{}.csv'.format(i), index=False)

            print "Partition {} training & prediction completed!".format(i)

        print "All completed!"

    def ensemble_all_partitions(self):

        # ensemble all the prediction results using each partition
        # get prediction file names list
        print "Ensemble all predictions:"
        prediction_files = []
        num_files = self.num_trees

        for i in range(num_files):
            prediction_files.append("submission_train{}.csv".format(i))

        # Read each files
        dfs = []
        for i in range(num_files):
            dfs.append(pd.read_csv(self.folder+prediction_files[i]))

        IDs = dfs[0]['TRIP_ID'].values
        Travel_Time = pd.DataFrame(columns=['TRAVEL_TIME'])

        for i in range(num_files):
            Travel_Time = pd.concat([Travel_Time['TRAVEL_TIME'], dfs[i]['TRAVEL_TIME']], axis=1)

        # Read the original test file
        df_test = pd.read_csv(self.folder+self.test_file)

        # Ensemble all results
        y_test =  np.maximum(np.mean(Travel_Time, axis=1), (df_test['Current_Snapshots']-1)*15)
        result = np.c_[IDs, np.around(y_test).astype(int)]
        df_result = pd.DataFrame(result, columns=['TRIP_ID', 'TRAVEL_TIME'])
        df_result.to_csv(self.folder+'submission_final.csv', index=False)

        print "Completed!"


def main():

    # step 1 - convert raw training data to sqlite database
    file_in = 'train.csv'
    file_out = 'train.sqlite'
    cts = CSVToSQL(FOLDER, file_in, file_out)
    cts.generate_sqlite()

    # step 2 - generate descriptive stats for train file
    file_in = 'train.sqlite'
    file_out = 'train_descriptive.sqlite'
    td = TrainDescriptive(FOLDER, file_in, file_out)
    td.transform()
    td.descriptive_hour()
    td.descriptive_weekday()
    td.descriptive_month()
    td.descriptive_driver()
    td.descriptive_stand()
    td.descriptive_caller()

    # step 3 - pre-process the training data
    file_in = 'train.csv'
    file_out = 'train_final.csv'
    tpp = TrainPreProcessing(folder=FOLDER, file_in=file_in, file_out=file_out)
    tpp.train_pre_process()
    tpp.fix_null()

    # step 4 - pre-process the test data
    file_in = 'test.csv'
    file_out = 'test_final.csv'
    tpp = TestPreProcessing(folder=FOLDER, file_in=file_in, file_out=file_out)
    tpp.test_pre_process()

    # step 5 - train the model with scalable random forest and predict for test set
    srf = ScalableRandomForest(num_trees=1000, q=0.75, folder=FOLDER,
                               train_file='train_final.csv', test_file='test_final.csv')
    srf.create_file_names()
    srf.split_train()
    srf.train_predict_by_partition()
    srf.ensemble_all_partitions()


if __name__ == '__main__':

    main()