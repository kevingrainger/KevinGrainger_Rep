import pandas as pd
import random


file_location = "C:/Users/kevin/Easyjet"

def fake_data():
    pilots = [{
        "CrewID": f"P{str(i).zfill(6)}",
        "CrewNames": ''.join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=5)) + " " + ''.join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=6)),
        "CrewAge": random.randint(20, 50),
        "Town": random.choice([f"Town-{str(i).zfill(2)}" for i in range(1, 51)])
    } for i in range(1, 10001)]

    cabin_crew = [{
        "CrewID": f"C{str(i).zfill(6)}",
        "CrewNames": ''.join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=5)) + " " + ''.join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=6)),
        "CrewAge": random.randint(20, 50),
        "Town": random.choice([f"Town-{str(i).zfill(2)}" for i in range(1, 51)])
    } for i in range(1, 50001)]

    df = pd.DataFrame(pilots + cabin_crew)
    df.to_csv(f"{file_location}/crews.csv", index=False)
    return df

def crew_stats():
    df = pd.read_csv(f"{file_location}/crews.csv")
    stats = df.groupby("Town")["CrewAge"].agg(
        Min_Age="min",
        Max_Age="max", 
        Average_Age="mean",
        Std_Dev_Age="std"
    ).reset_index()
    print(stats.to_string(index=False))
    stats.to_csv(f"{file_location}/crew_stats.csv", index=False)
    return stats

if __name__ == "__main__":
    fake_data()
    crew_stats()
