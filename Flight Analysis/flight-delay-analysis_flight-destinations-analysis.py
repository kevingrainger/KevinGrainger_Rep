import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#---------------------------------------------------------------------------
# Load Data

df = pd.read_csv("flights_with_aircrew.csv", low_memory=False)
print(f"Loaded {len(df)} flights")

# Make sure flight_date is in datetime format
df['flight_date'] = pd.to_datetime(df['departure_local_time']).dt.date
df['flight_date'] = pd.to_datetime(df['flight_date'])

# Convert codes to strings 
df['ORIGIN_AIRPORT'] = df['ORIGIN_AIRPORT'].astype(str)
df['DESTINATION_AIRPORT'] = df['DESTINATION_AIRPORT'].astype(str)

#---------------------------------------------------------------------------
# Flight Volume Over Time visual

flights_per_day = df.groupby('flight_date').size().reset_index(name='flight_count')
# Remove rows with zero flights and skip first day for clean grpah
flights_per_day = flights_per_day[flights_per_day['flight_date'] >= '2015-01-01']
flights_per_day = flights_per_day[flights_per_day['flight_count'] > 0]

plt.figure(figsize=(14, 6))
plt.plot(flights_per_day['flight_date'], flights_per_day['flight_count'], linewidth=2, color='blue')
plt.title('Flight Volume Over Time', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Number of Flights', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#---------------------------------------------------------------------------
# Most Popular Routes vis

df['route'] = df['ORIGIN_AIRPORT'] + ' → ' + df['DESTINATION_AIRPORT']
route_counts = df['route'].value_counts().head(20).reset_index()
route_counts.columns = ['route', 'count']

plt.figure(figsize=(12, 8))
plt.barh(route_counts['route'], route_counts['count'], color='steelblue')
plt.title('Most Popular Routes', fontsize=16)
plt.xlabel('Number of Flights', fontsize=12)
plt.ylabel('Route', fontsize=12)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

#---------------------------------------------------------------------------
print(f"\nTotal flights analyzed: {len(df):,}")
print(f"Date range: {df['flight_date'].min()} to {df['flight_date'].max()}")
print(f"\nMost popular route: {route_counts.iloc[0]['route']} ({route_counts.iloc[0]['count']} flights)")
