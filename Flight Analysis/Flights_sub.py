import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
#Pyspark was not working for me, I think it doesnt wok wiht my python version, I used Dask instead which I am not familar with apoloiges
if __name__ == '__main__':
    #dask stuff
    cluster = LocalCluster(n_workers=2, threads_per_worker=2)
    client = Client(cluster)
    crew_df = dd.read_csv("crews.csv")
    #having data type issues, so had to define each type
    flights_df = dd.read_csv("flights.csv", dtype={
        'AIR_TIME': 'float64',
        'ARRIVAL_DELAY': 'float64',
        'ARRIVAL_TIME': 'float64',
        'CANCELLATION_REASON': 'object', # was causing issues, probably needed to clean data
        'DEPARTURE_DELAY': 'float64',
        'DEPARTURE_TIME': 'float64',
        'ELAPSED_TIME': 'float64',
        'SCHEDULED_TIME': 'float64',
        'TAXI_IN': 'float64',
        'TAXI_OUT': 'float64',
        'WHEELS_OFF': 'float64',
        'WHEELS_ON': 'float64'
    }, low_memory=False)
    print("Loaded flights data")

    #crew types
    crew_df['CrewType'] = crew_df['CrewID'].apply(
        lambda x: 'Pilot' if str(x).startswith('P') else 'Cabin_Crew', #assigning func
        meta=('CrewType', 'object') #Dask data format
    )

    #flight times
    flights_df['DEPARTURE_TIME'] = flights_df['DEPARTURE_TIME'].fillna(0) #replacig missing values
    flights_df['ARRIVAL_TIME'] = flights_df['ARRIVAL_TIME'].fillna(0)
    flights_df['DEPARTURE_TIME_NUM'] = pd.to_numeric(flights_df['DEPARTURE_TIME'], errors='coerce').fillna(0).astype(int)#making uniform
    flights_df['ARRIVAL_TIME_NUM'] = pd.to_numeric(flights_df['ARRIVAL_TIME'], errors='coerce').fillna(0).astype(int)
    flights_df['DEPARTURE_TIME_STR'] = flights_df['DEPARTURE_TIME_NUM'].astype(str).str.zfill(4)#padding 
    flights_df['ARRIVAL_TIME_STR'] = flights_df['ARRIVAL_TIME_NUM'].astype(str).str.zfill(4)

    #geting correct format hours and mins
    def get_hour(time_str):
        return int(str(time_str)[:2]) % 24
    def get_minute(time_str):
        return int(str(time_str)[2:4]) % 60
    #applying these functions
    flights_df['dep_hour'] = flights_df['DEPARTURE_TIME_STR'].apply(get_hour, meta=('dep_hour', 'int64'))
    flights_df['dep_minute'] = flights_df['DEPARTURE_TIME_STR'].apply(get_minute, meta=('dep_minute', 'int64'))
    flights_df['arr_hour'] = flights_df['ARRIVAL_TIME_STR'].apply(get_hour, meta=('arr_hour', 'int64'))
    flights_df['arr_minute'] = flights_df['ARRIVAL_TIME_STR'].apply(get_minute, meta=('arr_minute', 'int64'))

    #Date in given form
    def create_datetime_string(year, month, day, hour, minute):
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d} {int(hour):02d}:{int(minute):02d}" #with zero padding
    #defining our data frames
    flights_df['dep_date_time'] = flights_df.apply(
        lambda row: create_datetime_string(row['YEAR'], row['MONTH'], row['DAY'], row['dep_hour'], row['dep_minute']),
        axis=1, meta=('dep_date_time', 'object')
    )
    flights_df['arr_date_time'] = flights_df.apply(
        lambda row: create_datetime_string(row['YEAR'], row['MONTH'], row['DAY'], row['arr_hour'], row['arr_minute']),
        axis=1, meta=('arr_date_time', 'object')
    )

    #timezones, with dask language, I used claude I am not very familar with this stuff
    flights_df['dep_utc'] = dd.to_datetime(flights_df['dep_date_time'])
    flights_df['arr_utc'] = dd.to_datetime(flights_df['arr_date_time'])
    flights_df['dep_local'] = flights_df['dep_utc'] - pd.Timedelta(hours=5)
    flights_df['arr_local'] = flights_df['arr_utc'] - pd.Timedelta(hours=8)
    flights_df['departure_local_time'] = flights_df['dep_local'].dt.strftime('%Y-%m-%d %H:%M')
    flights_df['arrival_local_time'] = flights_df['arr_local'].dt.strftime('%Y-%m-%d %H:%M')
    flights_df['flight_date'] = flights_df['dep_local'].dt.date

    print("Flight times cleaned and converted")
    
    print("Sample of the processed flights")
    sample = flights_df[['FLIGHT_NUMBER', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 
                        'departure_local_time', 'arrival_local_time']].head(5) #give first few lines
    print(sample)

    #crew lists
    crew_processed = crew_df.compute()
    pilots_list = crew_processed[crew_processed['CrewType'] == 'Pilot']['CrewID'].tolist()
    cabin_crew_list = crew_processed[crew_processed['CrewType'] == 'Cabin_Crew']['CrewID'].tolist()
    

    #assigning crew
    def assign_crew_rotation(df):
        df = df.sort_values(['flight_date', 'dep_utc']).reset_index(drop=True)
        
        df['pilot1_id'] = [pilots_list[(i * 2) % len(pilots_list)] for i in range(len(df))]
        df['pilot2_id'] = [pilots_list[(i * 2 + 1) % len(pilots_list)] for i in range(len(df))]#staggered to stop duplicates
        #cabin crew
        for j in range(5):
            df[f'cabin{j+1}_id'] = [cabin_crew_list[(i * 5 + j) % len(cabin_crew_list)] for i in range(len(df))] #function defined to change 'j' after each selection, meaning no duplicate picks
        return df

    #dataframe with crew columns
    new_df = flights_df._meta.copy()
    crew_columns = ['pilot1_id', 'pilot2_id', 'cabin1_id', 'cabin2_id', 'cabin3_id', 'cabin4_id', 'cabin5_id']
    for col in crew_columns:
        new_df[col] = pd.Series(dtype='object')

    
    print("assigning crews to flights")
    flights_with_crew = flights_df.map_partitions(assign_crew_rotation, meta=new_df)#parallel processing for Dask?
    print("Sample of crews:")
    sample_crew = flights_with_crew[['FLIGHT_NUMBER', 'pilot1_id', 'pilot2_id', 'cabin1_id', 'cabin2_id']].head(5)
    print(sample_crew)

    #sectors per aircraft per crew
    print("Computing sectors (will take a few mins)")
    flights_pd = flights_with_crew[['FLIGHT_NUMBER', 'TAIL_NUMBER'] + crew_columns].compute()
    

    crew_stats = []
    for crew_col in crew_columns: #cycle trhough cols, grouping by tail number
        temp = flights_pd.groupby(['TAIL_NUMBER', crew_col]).size().reset_index(name='total_sectors')#new data frame
        temp = temp.rename(columns={crew_col: 'CREW_ID'})
        crew_stats.append(temp)

    sectors_per_aircraft_crew = pd.concat(crew_stats).groupby(['TAIL_NUMBER', 'CREW_ID'])['total_sectors'].sum().reset_index()
    print("Sectors recorded")
    print("Saving files (wil take a few mins)")
    output_columns = ['YEAR', 'MONTH', 'DAY', 'AIRLINE', 'FLIGHT_NUMBER', 'TAIL_NUMBER',
                     'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'departure_local_time', 'arrival_local_time',
                     'pilot1_id', 'pilot2_id', 'cabin1_id', 'cabin2_id', 'cabin3_id', 'cabin4_id', 'cabin5_id']

    final_cols = [col for col in output_columns if col in flights_with_crew.columns] #only outputting ones that exisit and adding crew
    flights_with_crew[final_cols].to_csv("flights_with_aircrew.csv", index=False, single_file=True)
    print("Saved flights_with_aircrew.csv")

    client.close()
