# from flask import Flask, request, jsonify
# import pandas as pd
# import matplotlib.pyplot as plt
# import io
# import base64
# import numpy as np
# from datetime import datetime, timedelta

# app = Flask(__name__)

# # ----------------------------
# # Mock Sensor Data Generation
# # ----------------------------
# np.random.seed(42)

# buildings = [f"B{i}" for i in range(1, 21)]
# floors = [1,2,3,4,5]
# sensor_types = ["temperature", "pressure", "humidity"]

# # Generate timestamps for Aug 1-7, 2025 hourly
# timestamps = pd.date_range("2025-08-01", "2025-08-07 23:00:00", freq="H")

# rows = []
# for ts in timestamps:
#     for bld in buildings:
#         for fl in floors:
#             rows.append({
#                 "building": bld,
#                 "floor": fl,
#                 "timestamp": ts,
#                 "temperature": round(20 + 5*np.random.rand(),2),
#                 "pressure": round(990 + 30*np.random.rand(),2),
#                 "humidity": round(50 + 50*np.random.rand(),2)
#             })

# data = pd.DataFrame(rows)

# # ----------------------------
# # Utility: Generate Chart
# # ----------------------------
# def generate_chart(df, x_col, y_col, title, color='skyblue'):
#     plt.figure(figsize=(5,3))
#     df.plot(kind='bar', x=x_col, y=y_col, color=color, legend=False)
#     plt.title(title)
#     plt.ylabel(y_col.title())
#     plt.tight_layout()
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
#     plt.close()
#     buf.seek(0)
#     return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')

# # ----------------------------
# # Query API
# # ----------------------------
# @app.route("/api/query", methods=["POST"])
# def query():
#     req = request.get_json()
#     user_query = req.get("query", "").lower()

#     result = []
#     message = ""
#     image_base64 = None

#     # ----------------------------
#     # BASIC / FILTERING
#     # ----------------------------
#     if "average temperature" in user_query and "building 10" in user_query:
#         avg_temp = data[data["building"]=="B10"]["temperature"].mean()
#         message = f"Average temperature in Building 10 across all floors is {avg_temp:.2f} °C."
#         result = data[data["building"]=="B10"][["floor","temperature"]].to_dict(orient="records")
#         image_base64 = generate_chart(
#             data[data["building"]=="B10"].groupby("floor")["temperature"].mean().reset_index(),
#             "floor","temperature","Avg Temperature per Floor"
#         )

#     elif "highest pressure" in user_query:
#         row = data.loc[data["pressure"].idxmax()]
#         message = f"Highest pressure recorded: {row['pressure']} at {row['building']}-Floor {row['floor']}."
#         result = [row[["building","floor","pressure","timestamp"]].to_dict()]

#     elif "humidity readings above 80" in user_query:
#         filtered = data[data["humidity"]>80]
#         message = f"{len(filtered)} humidity readings above 80% found."
#         result = filtered[["building","floor","humidity","timestamp"]].to_dict(orient="records")

#     elif "lowest recorded temperature" in user_query and "building 16" in user_query:
#         df_b16 = data[data["building"]=="B16"]
#         row = df_b16.loc[df_b16["temperature"].idxmin()]
#         message = f"Lowest temperature in Building 16 is {row['temperature']} °C on floor {row['floor']}."
#         result = [row.to_dict()]

#     elif "total readings" in user_query:
#         counts = data[sensor_types].count().to_dict()
#         message = "Total readings per sensor type:"
#         result = [{"sensor_type":k,"count":v} for k,v in counts.items()]

#     # ----------------------------
#     # AGGREGATION / ANALYSIS
#     # ----------------------------
#     elif "average pressure for each building" in user_query:
#         df_avg = data.groupby("building")["pressure"].mean().reset_index()
#         message = "Average pressure per building:"
#         result = df_avg.to_dict(orient="records")
#         image_base64 = generate_chart(df_avg,"building","pressure","Average Pressure per Building")

#     elif "highest overall humidity in august" in user_query:
#         df_aug = data[data["timestamp"].dt.month==8]
#         df_avg = df_aug.groupby("building")["humidity"].mean()
#         building = df_avg.idxmax()
#         message = f"Building with highest overall humidity in August 2025: {building} ({df_avg.max():.2f}%)."
#         result = [{"building":building,"avg_humidity":df_avg.max()}]

#     elif "most temperature fluctuations" in user_query:
#         df_day = data.groupby(data["timestamp"].dt.date)["temperature"].agg(lambda x: x.max()-x.min())
#         day = df_day.idxmax()
#         message = f"Day with most temperature fluctuations: {day} ({df_day.max():.2f} °C)."
#         result = [{"date":str(day),"fluctuation":df_day.max()}]

#     elif "median value of all pressure sensors" in user_query:
#         median_pressure = data["pressure"].median()
#         message = f"Median pressure across all buildings: {median_pressure:.2f}"
#         result = [{"median_pressure":median_pressure}]

#     elif "most consistent temperature readings" in user_query:
#         floor_std = data.groupby(["building","floor"])["temperature"].std()
#         idx = floor_std.idxmin()
#         message = f"Floor with most consistent temperature: {idx} (std dev {floor_std.min():.2f})"
#         result = [{"building":idx[0],"floor":idx[1],"std_dev":floor_std.min()}]

#     # ----------------------------
#     # TREND / TIME-BASED
#     # ----------------------------
#     elif "trend of humidity readings" in user_query and "building 5" in user_query:
#         df_b5 = data[data["building"]=="B5"].set_index("timestamp")
#         df_plot = df_b5["humidity"].resample("D").mean()
#         message = "Trend of humidity readings for Building 5 over first week of August 2025."
#         result = [{"date":str(idx.date()),"avg_humidity":val} for idx,val in df_plot.items()]
#         # Chart
#         plt.figure(figsize=(5,3))
#         df_plot.plot(marker='o', color='green')
#         plt.ylabel("Humidity")
#         plt.title("Building 5 Humidity Trend")
#         plt.tight_layout()
#         buf = io.BytesIO()
#         plt.savefig(buf,format="png")
#         plt.close()
#         buf.seek(0)
#         image_base64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')

#     # ----------------------------
#     # ADVANCED / COMBINED
#     # ----------------------------
#     elif "correlations between temperature and humidity" in user_query:
#         df_corr = data.groupby("building")[["temperature","humidity"]].corr().iloc[0::2,-1].reset_index()
#         message = "Temperature vs Humidity correlation per building:"
#         result = df_corr[["building","humidity"]].to_dict(orient="records")

#     elif "compare average temperature on top floors vs bottom floors" in user_query:
#         df_top = data[data["floor"]==5]
#         df_bottom = data[data["floor"]==1]
#         avg_top = df_top.groupby("building")["temperature"].mean()
#         avg_bottom = df_bottom.groupby("building")["temperature"].mean()
#         message = "Comparison of average temperature: top floors vs bottom floors"
#         result = [{"building":b,"top_floor_avg":avg_top[b],"bottom_floor_avg":avg_bottom[b]} for b in avg_top.index]
#         image_base64 = generate_chart(pd.DataFrame(result),"building","top_floor_avg","Top Floor Temp")

#     # ----------------------------
#     # FALLBACK
#     # ----------------------------
#     else:
#         message = "I can answer questions about temperature, pressure, humidity, trends, and comparisons. Try asking one of these!"

#     return jsonify({
#         "message": message,
#         "result": result,
#         "image": image_base64
#     })

# @app.route("/health", methods=["GET"])
# def health():
#     return jsonify({"status":"ok"})

# if __name__ == "__main__":
#     app.run(debug=True)
from flask import Flask, request, jsonify
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# ----------------------------
# Mock Sensor Data Generation (in the correct 'long' format)
# ----------------------------
np.random.seed(42)

# Using a subset of buildings and a short time frame for lightweight mock data
buildings = [f"Building {i}" for i in [1, 5, 12, 16]] + [f"Building {i}" for i in range(2, 21)]
floors = [1, 2, 3, 4, 5]
sensor_types = ["temperature", "pressure", "humidity", "co2"]

# Generate timestamps for Aug 1-7, 2025 hourly
timestamps = pd.date_range("2025-08-01", "2025-08-07 23:00:00", freq="H")

rows = []
id_counter = 1
for ts in timestamps:
    for bld in buildings:
        for fl in floors:
            # Generate values for each sensor type
            temp = round(20 + 5 * np.random.rand(), 2)
            press = round(990 + 30 * np.random.rand(), 2)
            humid = round(50 + 50 * np.random.rand(), 2)
            co2_val = round(400 + 1000 * np.random.rand(), 2)

            # Convert to 'long' format rows (required for your sample data structure)
            for sensor_type, value, unit in [
                ("temperature", temp, "°C"),
                ("pressure", press, "hPa"),
                ("humidity", humid, "%"),
                ("co2", co2_val, "ppm")
            ]:
                rows.append({
                    "id": id_counter,
                    "building": bld,
                    "floor": fl,
                    "sensor_type": sensor_type,
                    "value": value,
                    "timestamp": ts,
                    "unit": unit,
                    # Added manufacturer and coordinates for completeness, though not used in logic
                    "manufacturer": np.random.choice(["Humidex", "ThermoSmart", "AirQualityPro", "SensTech"]),
                    "location_coordinates": f"53.{np.random.randint(330000, 360000)},-6.{np.random.randint(240000, 270000)}",
                    "sensor_status": np.random.choice(["active", "faulty", "maintenance"], p=[0.8, 0.1, 0.1])
                })
                id_counter += 1

data = pd.DataFrame(rows)

# Convert timestamp to datetime and floor to integer for correct filtering/comparison
data['timestamp'] = pd.to_datetime(data['timestamp'])
data['floor'] = data['floor'].astype(int)

# ----------------------------
# Utility: Generate Chart
# ----------------------------
def generate_chart(df, x_col, y_col, title, chart_type='bar', color='skyblue'):
    plt.figure(figsize=(8, 4))
    
    if chart_type == 'bar':
        df.plot(kind='bar', x=x_col, y=y_col, color=color, legend=False, ax=plt.gca())
    elif chart_type == 'line':
        # Assuming df has index as time for line charts
        plt.plot(df.index, df.values, marker='o', color=color)
    
    plt.title(title)
    plt.ylabel(y_col.title() if isinstance(y_col, str) else 'Value')
    plt.xlabel(x_col.title() if isinstance(x_col, str) else 'Index')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')

# ----------------------------
# Query API
# ----------------------------
@app.route("/api/query", methods=["POST"])
def query():
    req = request.get_json()
    user_query = req.get("query", "").lower()

    result = []
    message = ""
    image_base64 = None
    
    # Filter data for active sensors only by default (a reasonable assumption)
    df_active = data[data["sensor_status"] == "active"].copy() 

    # --- A. BASIC RETRIEVAL / FILTERING ---
    if "average temperature" in user_query and "building 10" in user_query:
        df_filtered = df_active[(df_active["building"]=="Building 10") & (df_active["sensor_type"]=="temperature")]
        avg_temp = df_filtered["value"].mean()
        message = f"Average temperature in Building 10 across all floors is {avg_temp:.2f} °C."
        df_chart = df_filtered.groupby("floor")["value"].mean().reset_index().rename(columns={"value":"temperature"})
        image_base64 = generate_chart(df_chart, "floor","temperature","Avg Temperature per Floor in Building 10")

    elif "highest pressure" in user_query:
        df_pressure = df_active[df_active["sensor_type"]=="pressure"]
        row = df_pressure.loc[df_pressure["value"].idxmax()]
        message = f"Highest pressure recorded: {row['value']} {row['unit']} at {row['building']}-Floor {row['floor']}."
        result = [row[["building","floor","value","timestamp"]].rename(columns={"value":"pressure"}).to_dict()]

    elif "humidity readings above" in user_query:
        df_humidity = df_active[df_active["sensor_type"]=="humidity"]
        # Find numeric condition (assuming 80% based on question)
        try:
            threshold = float(re.search(r'above\s*(\d+(\.\d+)?)', user_query).group(1))
        except:
            threshold = 80.0
            
        filtered = df_humidity[df_humidity["value"] > threshold]
        message = f"{len(filtered)} humidity readings above {threshold}% found."
        result = filtered[["building","floor","value","timestamp"]].rename(columns={"value":"humidity"}).to_dict(orient="records")

    elif "lowest recorded temperature" in user_query and "building 16" in user_query:
        df_b16_temp = df_active[(df_active["building"]=="Building 16") & (df_active["sensor_type"]=="temperature")]
        row = df_b16_temp.loc[df_b16_temp["value"].idxmin()]
        message = f"Lowest temperature in Building 16 is {row['value']} °C on floor {row['floor']}."
        result = [row.rename(columns={"value":"temperature"}).to_dict()]

    elif "total readings" in user_query and "sensor type" in user_query:
        counts = data.groupby("sensor_type")["id"].count().to_dict()
        message = "Total readings per sensor type (including inactive):"
        result = [{"sensor_type":k,"count":v} for k,v in counts.items()]

    # --- B. AGGREGATION / ANALYSIS ---
    elif "average pressure for each building" in user_query:
        df_pressure = df_active[df_active["sensor_type"]=="pressure"]
        df_avg = df_pressure.groupby("building")["value"].mean().reset_index().rename(columns={"value":"avg_pressure"})
        message = "Average pressure per building:"
        image_base64 = generate_chart(df_avg,"building","avg_pressure","Average Pressure per Building")
        result = df_avg.to_dict(orient="records")

    elif "highest overall humidity in august" in user_query:
        df_aug_humid = df_active[(df_active["timestamp"].dt.month==8) & (df_active["sensor_type"]=="humidity")]
        df_avg = df_aug_humid.groupby("building")["value"].mean()
        building = df_avg.idxmax()
        message = f"Building with highest overall humidity in August 2025: {building} ({df_avg.max():.2f}%)."
        result = [{"building":building,"avg_humidity":df_avg.max()}]

    elif "most temperature fluctuations" in user_query:
        df_temp = df_active[df_active["sensor_type"]=="temperature"]
        df_day = df_temp.groupby(df_temp["timestamp"].dt.date)["value"].agg(lambda x: x.max()-x.min())
        day = df_day.idxmax()
        message = f"Day with most temperature fluctuations: {day} ({df_day.max():.2f} °C)."
        result = [{"date":str(day),"fluctuation":df_day.max()}]

    elif "median value of all pressure sensors" in user_query:
        df_pressure = df_active[df_active["sensor_type"]=="pressure"]
        median_pressure = df_pressure["value"].median()
        message = f"Median pressure across all buildings: {median_pressure:.2f} hPa"
        result = [{"median_pressure":median_pressure}]

    elif "most consistent temperature readings" in user_query:
        df_temp = df_active[df_active["sensor_type"]=="temperature"]
        floor_std = df_temp.groupby(["building","floor"])["value"].std()
        idx = floor_std.idxmin()
        message = f"Location with most consistent temperature: {idx[0]}-Floor {idx[1]} (Std Dev: {floor_std.min():.2f} °C)"
        result = [{"building":idx[0],"floor":idx[1],"std_dev":floor_std.min()}]

    # --- C. TREND / TIME-BASED ---
    elif "trend of humidity readings" in user_query and "building 5" in user_query:
        df_b5_humid = df_active[(df_active["building"]=="Building 5") & (df_active["sensor_type"]=="humidity")]
        df_plot = df_b5_humid.set_index("timestamp")["value"].resample("D").mean()
        message = "Trend of humidity readings for Building 5 over first week of August 2025."
        
        df_chart = df_plot.rename("Avg Humidity") # Rename for plotting utility
        image_base64 = generate_chart(df_chart, "timestamp", "Avg Humidity", "Building 5 Humidity Trend", chart_type='line', color='green')
        result = [{"date":str(idx.date()),"avg_humidity":val} for idx,val in df_plot.items()]

    elif "largest change in pressure in a single day" in user_query:
        df_pressure = df_active[df_active["sensor_type"]=="pressure"]
        df_day_range = df_pressure.groupby(["building", df_pressure["timestamp"].dt.date])["value"].agg(lambda x: x.max() - x.min())
        idx = df_day_range.idxmax()
        message = f"Building/Day with largest change in pressure: **{idx[0]}** on **{idx[1]}** ({df_day_range.max():.2f} hPa)."
        result = [{"building":idx[0],"date":str(idx[1]),"max_change":df_day_range.max()}]

    elif "hours of the day when temperature is generally highest" in user_query:
        df_temp = df_active[df_active["sensor_type"]=="temperature"]
        df_hour_avg = df_temp.groupby(df_temp["timestamp"].dt.hour)["value"].mean()
        hour = df_hour_avg.idxmax()
        message = f"Temperature is generally highest at hour **{hour:02d}:00** UTC ({df_hour_avg.max():.2f} °C)."
        
        df_chart = df_hour_avg.reset_index().rename(columns={"timestamp":"Hour", "value":"Avg Temperature"})
        image_base64 = generate_chart(df_chart, "Hour", "Avg Temperature", "Avg Temperature by Hour of Day", color='orange')
        result = [{"hour":f"{h:02d}:00","avg_temp":t} for h,t in df_hour_avg.items()]
        
    elif "temperature increased consecutively for at least 3 hours" in user_query:
        df_temp = df_active[df_active["sensor_type"]=="temperature"].sort_values("timestamp")
        result_buildings = set()

        for bld, group in df_temp.groupby("building"):
            group['temp_diff'] = group['value'].diff()
            group['is_increase'] = (group['temp_diff'] > 0).astype(int)
            
            # Identify consecutive increases (using a rolling sum trick)
            group['consec_increase'] = group['is_increase'].rolling(window=3).sum()
            
            if group['consec_increase'].max() >= 3:
                result_buildings.add(bld)
        
        buildings_list = list(result_buildings)
        message = f"Buildings with temperature increasing consecutively for at least 3 hours: **{', '.join(buildings_list)}**"
        result = [{"building":b} for b in buildings_list]
    
    elif "outliers in the pressure readings" in user_query:
        df_pressure = df_active[df_active["sensor_type"]=="pressure"].copy()
        df_pressure['z_score'] = zscore(df_pressure['value'])
        outliers = df_pressure[abs(df_pressure['z_score']) > 3]
        
        message = f"**{len(outliers)}** pressure outliers detected (Z-score > 3 or < -3)."
        result = outliers[["building", "floor", "value", "timestamp", "z_score"]].rename(columns={"value":"pressure"}).to_dict(orient="records")

    # --- D. COMBINED / ADVANCED QUESTIONS ---
    elif "correlations between temperature and humidity" in user_query:
        # Pivot to wide format to get temp and humid side-by-side for correlation
        df_wide = df_active.pivot_table(index=['building', 'timestamp'], columns='sensor_type', values='value').reset_index()
        df_corr = df_wide.groupby("building")[["temperature", "humidity"]].corr().unstack().iloc[:, 1].reset_index().rename(columns={"humidity":"correlation"})
        message = "Temperature vs Humidity Pearson **correlation** per building:"
        result = df_corr[["building","correlation"]].to_dict(orient="records")

    elif "anomalies where pressure is unusually low or high" in user_query:
        df_pressure = df_active[df_active["sensor_type"]=="pressure"].copy()
        # Group by building and calculate mean and std dev for local anomaly detection
        stats = df_pressure.groupby("building")["value"].agg(['mean', 'std'])
        
        anomalies = []
        for bld, group in df_pressure.groupby("building"):
            mean = stats.loc[bld, 'mean']
            std = stats.loc[bld, 'std']
            # Find values outside 2 standard deviations (a common anomaly threshold)
            anom_bld = group[(group['value'] > mean + 2*std) | (group['value'] < mean - 2*std)]
            if not anom_bld.empty:
                anomalies.extend(anom_bld.rename(columns={"value":"pressure"}).to_dict(orient="records"))

        message = f"**{len(anomalies)}** local pressure anomalies detected (outside 2 local standard deviations)."
        result = anomalies

    elif "consistently have higher humidity than average" in user_query:
        df_humidity = df_active[df_active["sensor_type"]=="humidity"]
        overall_avg_humid = df_humidity["value"].mean()
        df_avg_humid = df_humidity.groupby("building")["value"].mean().reset_index().rename(columns={"value":"avg_humidity"})
        df_high = df_avg_humid[df_avg_humid["avg_humidity"] > overall_avg_humid]
        
        message = f"Buildings consistently having **higher humidity** than average ({overall_avg_humid:.2f}%): **{', '.join(df_high['building'].tolist())}**"
        
        image_base64 = generate_chart(df_high,"building","avg_humidity","Buildings with Higher-Than-Average Humidity")
        result = df_high.to_dict(orient="records")

    elif "compare average temperature on top floors vs bottom floors" in user_query:
        df_temp = df_active[df_active["sensor_type"]=="temperature"]
        max_floor = df_temp['floor'].max()
        
        df_top = df_temp[df_temp["floor"]==max_floor]
        df_bottom = df_temp[df_temp["floor"]==1]
        
        avg_top = df_top.groupby("building")["value"].mean()
        avg_bottom = df_bottom.groupby("building")["value"].mean()
        
        comparison = pd.DataFrame({'top_floor_avg': avg_top, 'bottom_floor_avg': avg_bottom}).reset_index()
        message = f"Comparison of average temperature: top floors (Floor {max_floor}) vs bottom floors (Floor 1)"
        
        # Plot top floor temps for a visualization
        image_base64 = generate_chart(comparison,"building","top_floor_avg",f"Average Temperature on Top Floors (Floor {max_floor})")
        result = comparison.to_dict(orient="records")

    elif "predict the next hour’s temperature for building 12-floor 2" in user_query:
        bld = "Building 12"
        fl = 2
        df_target = df_active[(df_active["building"]==bld) & (df_active["floor"]==fl) & (df_active["sensor_type"]=="temperature")].sort_values("timestamp")
        
        # Prepare data for Linear Regression
        df_target = df_target.reset_index(drop=True)
        start_time = df_target['timestamp'].min()
        df_target['time_sec'] = (df_target['timestamp'] - start_time).dt.total_seconds()
        
        X = df_target[['time_sec']]
        y = df_target['value']
        
        # Train the model
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict the next hour (3600 seconds after the last reading)
        last_time_sec = df_target['time_sec'].max()
        next_time_sec = last_time_sec + 3600
        next_temp_pred = model.predict([[next_time_sec]])[0]
        
        message = f"Predicted next hour's temperature for **{bld}-Floor {fl}** is: **{next_temp_pred:.2f} °C** (based on simple linear trend)."
        result = [{"prediction_time": (df_target['timestamp'].max() + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"), 
                   "predicted_temperature": next_temp_pred}]

    # ----------------------------
    # FALLBACK
    # ----------------------------
    else:
        message = f"Query not fully implemented or recognized. I received: '{user_query}'"

    return jsonify({
        "message": message,
        "result": result,
        "image": image_base64
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"ok"})

if __name__ == "__main__":
    import re # Used for regex in advanced query parsing (humidity threshold)
    app.run(debug=True, port=3030)