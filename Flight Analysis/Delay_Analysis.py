import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

flights_df = pd.read_csv("flights.csv", low_memory=False)
crews_df = pd.read_csv("crews.csv")

# Add this - needed for month_name and season columns later
flights_df['date'] = pd.to_datetime(flights_df[['YEAR', 'MONTH', 'DAY']])
flights_df['month_name'] = flights_df['date'].dt.month_name()
flights_df['season'] = flights_df['MONTH'].apply(lambda x: 'Winter' if x in [12,1,2] else 'Spring' if x in [3,4,5] else 'Summer' if x in [6,7,8] else 'Autumn')

# ============================================================================
# Delay Analysis - by Airport

#for each airport, calculate avg delay, median delay, total flights, and % delayed flights
#first create a column to mark if flight was delayed >15 minutes
flights_df['is_delayed_dep'] = (flights_df['DEPARTURE_DELAY'] > 15).astype(int)
flights_df['is_delayed_arr'] = (flights_df['ARRIVAL_DELAY'] > 15).astype(int)

#group flights by airport
grouped = flights_df.groupby('ORIGIN_AIRPORT')
#calculate average delay for each airport
avg_delay = grouped['DEPARTURE_DELAY'].mean()
#calculate median delay for each airport
median_delay = grouped['DEPARTURE_DELAY'].median()
#count how many flights were delayed at each airport
delayed_flights = grouped['is_delayed_dep'].sum()
#count total flights at each airport
total_flights = grouped['FLIGHT_NUMBER'].count()
#combine all statistics into one dataframe
airport_dep_delay = pd.DataFrame({
    'avg_delay': avg_delay,
    'median_delay': median_delay,
    'delayed_flights': delayed_flights,
    'total_flights': total_flights
}).round(2)

#calculate percentage of flights that were delayed
airport_dep_delay['delay_rate'] = (airport_dep_delay['delayed_flights'] / airport_dep_delay['total_flights'] * 100).round(1)
#filter airports with at least 1000 flights
airport_dep_delay = airport_dep_delay[airport_dep_delay['total_flights'] >= 1000]
#sort by delay rate %
airport_dep_delay = airport_dep_delay.sort_values('delay_rate', ascending=False)

# bar chart showing delay rate by airport
plt.figure(figsize=(14, 8))
top_delay_airports = airport_dep_delay.head(15)
plt.barh(range(len(top_delay_airports)), top_delay_airports['delay_rate'], color='coral', alpha=0.8)

plt.yticks(range(len(top_delay_airports)), top_delay_airports.index)
plt.xlabel('Delay Rate (% delayed >15 min)', fontsize=12)
plt.ylabel('Airport', fontsize=12)
plt.title('Airports by Delay Rate', fontsize=14)
plt.gca().invert_yaxis()  #invert so highest is at top
plt.tight_layout()
plt.show()

# ============================================================================
# Delay Causes

delay_columns = ['AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 
                 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY']

# Sum total minutes for each delay cause
delay_totals = {}
total_delay_minutes = sum(delay_totals.values())
delay_df = pd.DataFrame(list(delay_totals.items()), columns=['Cause', 'Minutes'])
#sort by minutes column in descending
delay_df = delay_df.sort_values('Minutes', ascending=False)

#Pie chart of delay causes
plt.figure(figsize=(10, 8))
plt.pie(delay_totals.values(), labels=delay_totals.keys(), autopct='%1.1f%%',
        startangle=90)
plt.title('Delay Causes', fontsize=14)
plt.tight_layout()
plt.show() 

# ============================================================================
# Delay by season & month
# Calculate average delays grouped by season
season_delays = flights_df.groupby('season').agg({
    'DEPARTURE_DELAY': 'mean',
    'ARRIVAL_DELAY': 'mean'
}).round(2)

print("\nDelays by Season:")

#average delays by month, reorder chronologically
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']
month_delays = flights_df.groupby('month_name').agg({
    'DEPARTURE_DELAY': 'mean',
    'ARRIVAL_DELAY': 'mean'
}).round(2)
#reindex() to reorder, prolly a lazy fix
month_delays_sorted = month_delays.reindex(month_order)

#plot showing delays per month
plt.figure(figsize=(14, 6))
x = range(len(month_delays_sorted))
plt.plot(x, month_delays_sorted['DEPARTURE_DELAY'], marker='o', label='Departure Delay', linewidth=2)
plt.plot(x, month_delays_sorted['ARRIVAL_DELAY'], marker='s', label='Arrival Delay', linewidth=2)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Average Delay (mins)', fontsize=12)
plt.title('Average Flight Delays by Month', fontsize=14)
plt.xticks(x, month_delays_sorted.index, rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show() 

# ============================================================================
# Aircraft use

#missing some analysis apolgies 
#value_counts()- how many flights each aircraft (tail number)
aircraft_flights = flights_df['TAIL_NUMBER'].value_counts()
print(f"\nAircraft: {len(aircraft_flights):,} total, avg {aircraft_flights.mean():.1f} flights each")
print(f"Median flights per aircraft: {aircraft_flights.median():.1f}")
print(f"Most active aircraft: {aircraft_flights.index[0]} ({aircraft_flights.iloc[0]:,} flights)")
print(f"Least active aircraft: {aircraft_flights.index[-1]} ({aircraft_flights.iloc[-1]:,} flights)")

#Histogram showing distribution of flights per aircraft
plt.figure(figsize=(12, 6))
plt.hist(aircraft_flights, bins=50, color='mediumseagreen', edgecolor='black', alpha=0.7)
plt.xlabel('Number of Flights', fontsize=12)
plt.ylabel('Number of Aircraft', fontsize=12)
plt.title('Aircraft Use Distribution', fontsize=14)
plt.axvline(aircraft_flights.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {aircraft_flights.mean():.1f}')
plt.legend()
plt.tight_layout()
plt.show()