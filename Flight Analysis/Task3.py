import pandas as pd
import numpy as np


flights_df = pd.read_csv("flights_with_aircrew.csv", low_memory=False)
crews_df = pd.read_csv("crews.csv")

print("Available columns:", flights_df.columns.tolist())

#process dates for season data
flights_df['date'] = pd.to_datetime(flights_df[['YEAR', 'MONTH', 'DAY']])
flights_df['month_name'] = flights_df['date'].dt.month_name()
flights_df['season'] = flights_df['MONTH'].apply(lambda x: 'Winter' if x in [12,1,2] else 'Spring' if x in [3,4,5] else 'Summer' if x in [6,7,8] else 'Fall')


#flights by month
monthly = flights_df.groupby('month_name').size().reset_index(name='flights')
monthly['pct'] = (monthly['flights'] / len(flights_df) * 100).round(1)
print("Flights by month:")
for _, row in monthly.iterrows():
    print(f"{row['month_name']}: {row['flights']:,} ({row['pct']}%)")

#flights by season
seasonal = flights_df.groupby('season').size().reset_index(name='flights')
seasonal['pct'] = (seasonal['flights'] / len(flights_df) * 100).round(1)
print("Flights by season:")
for _, row in seasonal.iterrows():
    print(f"{row['season']}: {row['flights']:,} ({row['pct']}%)")


#Top airports
origin_counts = flights_df['ORIGIN_AIRPORT'].value_counts()
print("Top 10 Origin Airports:")
for i, (airport, count) in enumerate(origin_counts.head(10).items(), 1):
    pct = (count / len(flights_df) * 100)
    print(f"{i:2d}. {airport}: {count:,} ({pct:.1f}%)")#formatted

#Crew
pilot_ages = crews_df[crews_df['CrewID'].str.startswith('P')]['CrewAge']
cabin_ages = crews_df[crews_df['CrewID'].str.startswith('C')]['CrewAge']
print(f"Pilots: Age {pilot_ages.min()}-{pilot_ages.max()}, Avg: {pilot_ages.mean():.1f}")
print(f"Cabin crew: Age {cabin_ages.min()}-{cabin_ages.max()}, Avg: {cabin_ages.mean():.1f}")

#missing some analysis apolgies 

#aircraft
aircraft_flights = flights_df['TAIL_NUMBER'].value_counts()
print(f"\nAircraft: {len(aircraft_flights):,} total, avg {aircraft_flights.mean():.1f} flights each")
