# from flask import Flask, request, jsonify
# import pandas as pd
# import matplotlib.pyplot as plt
# import io
# import base64
# import numpy as np
# from datetime import datetime, timedelta
# from scipy.stats import zscore
# from sklearn.linear_model import LinearRegression
# import requests
# app = Flask(__name__)

# # ----------------------------
# # Mock Sensor Data Generation (in the correct 'long' format)
# # ----------------------------
# np.random.seed(42)
# def get_bike_data():
#     url = "https://api.cyclocity.fr/contracts/dublin/gbfs/station_status.json"
#     response = requests.get(url)
#     return response.json()
# # Using a subset of buildings and a short time frame for lightweight mock data
# buildings = [f"Building {i}" for i in [1, 5, 12, 16]] + [f"Building {i}" for i in range(2, 21)]
# floors = [1, 2, 3, 4, 5]
# sensor_types = ["temperature", "pressure", "humidity", "co2"]

# # Generate timestamps for Aug 1-7, 2025 hourly
# timestamps = pd.date_range("2025-08-01", "2025-08-07 23:00:00", freq="H")

# rows = []
# id_counter = 1
# for ts in timestamps:
#     for bld in buildings:
#         for fl in floors:
#             # Generate values for each sensor type
#             temp = round(20 + 5 * np.random.rand(), 2)
#             press = round(990 + 30 * np.random.rand(), 2)
#             humid = round(50 + 50 * np.random.rand(), 2)
#             co2_val = round(400 + 1000 * np.random.rand(), 2)

#             # Convert to 'long' format rows (required for your sample data structure)
#             for sensor_type, value, unit in [
#                 ("temperature", temp, "°C"),
#                 ("pressure", press, "hPa"),
#                 ("humidity", humid, "%"),
#                 ("co2", co2_val, "ppm")
#             ]:
#                 rows.append({
#                     "id": id_counter,
#                     "building": bld,
#                     "floor": fl,
#                     "sensor_type": sensor_type,
#                     "value": value,
#                     "timestamp": ts,
#                     "unit": unit,
#                     # Added manufacturer and coordinates for completeness, though not used in logic
#                     "manufacturer": np.random.choice(["Humidex", "ThermoSmart", "AirQualityPro", "SensTech"]),
#                     "location_coordinates": f"53.{np.random.randint(330000, 360000)},-6.{np.random.randint(240000, 270000)}",
#                     "sensor_status": np.random.choice(["active", "faulty", "maintenance"], p=[0.8, 0.1, 0.1])
#                 })
#                 id_counter += 1

# data = pd.DataFrame(rows)

# # Convert timestamp to datetime and floor to integer for correct filtering/comparison
# data['timestamp'] = pd.to_datetime(data['timestamp'])
# data['floor'] = data['floor'].astype(int)

# # ----------------------------
# # Utility: Generate Chart
# # ----------------------------
# def generate_chart(df, x_col, y_col, title, chart_type='bar', color='skyblue'):
#     plt.figure(figsize=(8, 4))
    
#     if chart_type == 'bar':
#         df.plot(kind='bar', x=x_col, y=y_col, color=color, legend=False, ax=plt.gca())
#     elif chart_type == 'line':
#         # Assuming df has index as time for line charts
#         plt.plot(df.index, df.values, marker='o', color=color)
    
#     plt.title(title)
#     plt.ylabel(y_col.title() if isinstance(y_col, str) else 'Value')
#     plt.xlabel(x_col.title() if isinstance(x_col, str) else 'Index')
#     plt.xticks(rotation=45, ha='right')
#     plt.grid(axis='y', linestyle='--')
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
    
#     # Filter data for active sensors only by default (a reasonable assumption)
#     df_active = data[data["sensor_status"] == "active"].copy() 

#     # --- A. BASIC RETRIEVAL / FILTERING ---
#     if "average temperature" in user_query and "building 10" in user_query:
#         df_filtered = df_active[(df_active["building"]=="Building 10") & (df_active["sensor_type"]=="temperature")]
#         avg_temp = df_filtered["value"].mean()
#         message = f"Average temperature in Building 10 across all floors is {avg_temp:.2f} °C."
#         df_chart = df_filtered.groupby("floor")["value"].mean().reset_index().rename(columns={"value":"temperature"})
#         image_base64 = generate_chart(df_chart, "floor","temperature","Avg Temperature per Floor in Building 10")

#     elif "highest pressure" in user_query:
#         df_pressure = df_active[df_active["sensor_type"]=="pressure"]
#         row = df_pressure.loc[df_pressure["value"].idxmax()]
#         message = f"Highest pressure recorded: {row['value']} {row['unit']} at {row['building']}-Floor {row['floor']}."
#         result = [row[["building","floor","value","timestamp"]].rename(columns={"value":"pressure"}).to_dict()]

#     elif "humidity readings above" in user_query:
#         df_humidity = df_active[df_active["sensor_type"]=="humidity"]
#         # Find numeric condition (assuming 80% based on question)
#         try:
#             threshold = float(re.search(r'above\s*(\d+(\.\d+)?)', user_query).group(1))
#         except:
#             threshold = 80.0
            
#         filtered = df_humidity[df_humidity["value"] > threshold]
#         message = f"{len(filtered)} humidity readings above {threshold}% found."
#         result = filtered[["building","floor","value","timestamp"]].rename(columns={"value":"humidity"}).to_dict(orient="records")

#     elif "lowest recorded temperature" in user_query and "building 16" in user_query:
#         df_b16_temp = df_active[(df_active["building"]=="Building 16") & (df_active["sensor_type"]=="temperature")]
#         row = df_b16_temp.loc[df_b16_temp["value"].idxmin()]
#         message = f"Lowest temperature in Building 16 is {row['value']} °C on floor {row['floor']}."
#         result = [row.rename(columns={"value":"temperature"}).to_dict()]

#     elif "total readings" in user_query and "sensor type" in user_query:
#         counts = data.groupby("sensor_type")["id"].count().to_dict()
#         message = "Total readings per sensor type (including inactive):"
#         result = [{"sensor_type":k,"count":v} for k,v in counts.items()]

#     # --- B. AGGREGATION / ANALYSIS ---
#     elif "average pressure for each building" in user_query:
#         df_pressure = df_active[df_active["sensor_type"]=="pressure"]
#         df_avg = df_pressure.groupby("building")["value"].mean().reset_index().rename(columns={"value":"avg_pressure"})
#         message = "Average pressure per building:"
#         image_base64 = generate_chart(df_avg,"building","avg_pressure","Average Pressure per Building")
#         result = df_avg.to_dict(orient="records")

#     elif "highest overall humidity in august" in user_query:
#         df_aug_humid = df_active[(df_active["timestamp"].dt.month==8) & (df_active["sensor_type"]=="humidity")]
#         df_avg = df_aug_humid.groupby("building")["value"].mean()
#         building = df_avg.idxmax()
#         message = f"Building with highest overall humidity in August 2025: {building} ({df_avg.max():.2f}%)."
#         result = [{"building":building,"avg_humidity":df_avg.max()}]

#     elif "most temperature fluctuations" in user_query:
#         df_temp = df_active[df_active["sensor_type"]=="temperature"]
#         df_day = df_temp.groupby(df_temp["timestamp"].dt.date)["value"].agg(lambda x: x.max()-x.min())
#         day = df_day.idxmax()
#         message = f"Day with most temperature fluctuations: {day} ({df_day.max():.2f} °C)."
#         result = [{"date":str(day),"fluctuation":df_day.max()}]

#     elif "median value of all pressure sensors" in user_query:
#         df_pressure = df_active[df_active["sensor_type"]=="pressure"]
#         median_pressure = df_pressure["value"].median()
#         message = f"Median pressure across all buildings: {median_pressure:.2f} hPa"
#         result = [{"median_pressure":median_pressure}]

#     elif "most consistent temperature readings" in user_query:
#         df_temp = df_active[df_active["sensor_type"]=="temperature"]
#         floor_std = df_temp.groupby(["building","floor"])["value"].std()
#         idx = floor_std.idxmin()
#         message = f"Location with most consistent temperature: {idx[0]}-Floor {idx[1]} (Std Dev: {floor_std.min():.2f} °C)"
#         result = [{"building":idx[0],"floor":idx[1],"std_dev":floor_std.min()}]

#     # --- C. TREND / TIME-BASED ---
#     elif "trend of humidity readings" in user_query and "building 5" in user_query:
#         df_b5_humid = df_active[(df_active["building"]=="Building 5") & (df_active["sensor_type"]=="humidity")]
#         df_plot = df_b5_humid.set_index("timestamp")["value"].resample("D").mean()
#         message = "Trend of humidity readings for Building 5 over first week of August 2025."
        
#         df_chart = df_plot.rename("Avg Humidity") # Rename for plotting utility
#         image_base64 = generate_chart(df_chart, "timestamp", "Avg Humidity", "Building 5 Humidity Trend", chart_type='line', color='green')
#         result = [{"date":str(idx.date()),"avg_humidity":val} for idx,val in df_plot.items()]

#     elif "largest change in pressure in a single day" in user_query:
#         df_pressure = df_active[df_active["sensor_type"]=="pressure"]
#         df_day_range = df_pressure.groupby(["building", df_pressure["timestamp"].dt.date])["value"].agg(lambda x: x.max() - x.min())
#         idx = df_day_range.idxmax()
#         message = f"Building/Day with largest change in pressure: **{idx[0]}** on **{idx[1]}** ({df_day_range.max():.2f} hPa)."
#         result = [{"building":idx[0],"date":str(idx[1]),"max_change":df_day_range.max()}]

#     elif "hours of the day when temperature is generally highest" in user_query:
#         df_temp = df_active[df_active["sensor_type"]=="temperature"]
#         df_hour_avg = df_temp.groupby(df_temp["timestamp"].dt.hour)["value"].mean()
#         hour = df_hour_avg.idxmax()
#         message = f"Temperature is generally highest at hour **{hour:02d}:00** UTC ({df_hour_avg.max():.2f} °C)."
        
#         df_chart = df_hour_avg.reset_index().rename(columns={"timestamp":"Hour", "value":"Avg Temperature"})
#         image_base64 = generate_chart(df_chart, "Hour", "Avg Temperature", "Avg Temperature by Hour of Day", color='orange')
#         result = [{"hour":f"{h:02d}:00","avg_temp":t} for h,t in df_hour_avg.items()]
        
#     elif "temperature increased consecutively for at least 3 hours" in user_query:
#         df_temp = df_active[df_active["sensor_type"]=="temperature"].sort_values("timestamp")
#         result_buildings = set()

#         for bld, group in df_temp.groupby("building"):
#             group['temp_diff'] = group['value'].diff()
#             group['is_increase'] = (group['temp_diff'] > 0).astype(int)
            
#             # Identify consecutive increases (using a rolling sum trick)
#             group['consec_increase'] = group['is_increase'].rolling(window=3).sum()
            
#             if group['consec_increase'].max() >= 3:
#                 result_buildings.add(bld)
        
#         buildings_list = list(result_buildings)
#         message = f"Buildings with temperature increasing consecutively for at least 3 hours: **{', '.join(buildings_list)}**"
#         result = [{"building":b} for b in buildings_list]
    
#     elif "outliers in the pressure readings" in user_query:
#         df_pressure = df_active[df_active["sensor_type"]=="pressure"].copy()
#         df_pressure['z_score'] = zscore(df_pressure['value'])
#         outliers = df_pressure[abs(df_pressure['z_score']) > 3]
        
#         message = f"**{len(outliers)}** pressure outliers detected (Z-score > 3 or < -3)."
#         result = outliers[["building", "floor", "value", "timestamp", "z_score"]].rename(columns={"value":"pressure"}).to_dict(orient="records")

#     # --- D. COMBINED / ADVANCED QUESTIONS ---
#     elif "correlations between temperature and humidity" in user_query:
#         # Pivot to wide format to get temp and humid side-by-side for correlation
#         df_wide = df_active.pivot_table(index=['building', 'timestamp'], columns='sensor_type', values='value').reset_index()
#         df_corr = df_wide.groupby("building")[["temperature", "humidity"]].corr().unstack().iloc[:, 1].reset_index().rename(columns={"humidity":"correlation"})
#         message = "Temperature vs Humidity Pearson **correlation** per building:"
#         result = df_corr[["building","correlation"]].to_dict(orient="records")

#     elif "anomalies where pressure is unusually low or high" in user_query:
#         df_pressure = df_active[df_active["sensor_type"]=="pressure"].copy()
#         # Group by building and calculate mean and std dev for local anomaly detection
#         stats = df_pressure.groupby("building")["value"].agg(['mean', 'std'])
        
#         anomalies = []
#         for bld, group in df_pressure.groupby("building"):
#             mean = stats.loc[bld, 'mean']
#             std = stats.loc[bld, 'std']
#             # Find values outside 2 standard deviations (a common anomaly threshold)
#             anom_bld = group[(group['value'] > mean + 2*std) | (group['value'] < mean - 2*std)]
#             if not anom_bld.empty:
#                 anomalies.extend(anom_bld.rename(columns={"value":"pressure"}).to_dict(orient="records"))

#         message = f"**{len(anomalies)}** local pressure anomalies detected (outside 2 local standard deviations)."
#         result = anomalies

#     elif "consistently have higher humidity than average" in user_query:
#         df_humidity = df_active[df_active["sensor_type"]=="humidity"]
#         overall_avg_humid = df_humidity["value"].mean()
#         df_avg_humid = df_humidity.groupby("building")["value"].mean().reset_index().rename(columns={"value":"avg_humidity"})
#         df_high = df_avg_humid[df_avg_humid["avg_humidity"] > overall_avg_humid]
        
#         message = f"Buildings consistently having **higher humidity** than average ({overall_avg_humid:.2f}%): **{', '.join(df_high['building'].tolist())}**"
        
#         image_base64 = generate_chart(df_high,"building","avg_humidity","Buildings with Higher-Than-Average Humidity")
#         result = df_high.to_dict(orient="records")

#     elif "compare average temperature on top floors vs bottom floors" in user_query:
#         df_temp = df_active[df_active["sensor_type"]=="temperature"]
#         max_floor = df_temp['floor'].max()
        
#         df_top = df_temp[df_temp["floor"]==max_floor]
#         df_bottom = df_temp[df_temp["floor"]==1]
        
#         avg_top = df_top.groupby("building")["value"].mean()
#         avg_bottom = df_bottom.groupby("building")["value"].mean()
        
#         comparison = pd.DataFrame({'top_floor_avg': avg_top, 'bottom_floor_avg': avg_bottom}).reset_index()
#         message = f"Comparison of average temperature: top floors (Floor {max_floor}) vs bottom floors (Floor 1)"
        
#         # Plot top floor temps for a visualization
#         image_base64 = generate_chart(comparison,"building","top_floor_avg",f"Average Temperature on Top Floors (Floor {max_floor})")
#         result = comparison.to_dict(orient="records")

#     elif "predict the next hour’s temperature for building 12-floor 2" in user_query:
#         bld = "Building 12"
#         fl = 2
#         df_target = df_active[(df_active["building"]==bld) & (df_active["floor"]==fl) & (df_active["sensor_type"]=="temperature")].sort_values("timestamp")
        
#         # Prepare data for Linear Regression
#         df_target = df_target.reset_index(drop=True)
#         start_time = df_target['timestamp'].min()
#         df_target['time_sec'] = (df_target['timestamp'] - start_time).dt.total_seconds()
        
#         X = df_target[['time_sec']]
#         y = df_target['value']
        
#         # Train the model
#         model = LinearRegression()
#         model.fit(X, y)
        
#         # Predict the next hour (3600 seconds after the last reading)
#         last_time_sec = df_target['time_sec'].max()
#         next_time_sec = last_time_sec + 3600
#         next_temp_pred = model.predict([[next_time_sec]])[0]
        
#         message = f"Predicted next hour's temperature for **{bld}-Floor {fl}** is: **{next_temp_pred:.2f} °C** (based on simple linear trend)."
#         result = [{"prediction_time": (df_target['timestamp'].max() + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"), 
#                    "predicted_temperature": next_temp_pred}]

#     # ----------------------------
#     # FALLBACK
#     # ----------------------------
#     else:
#         message = f"Query not fully implemented or recognized. I received: '{user_query}'"

#     return jsonify({
#         "message": message,
#         "result": result,
#         "image": image_base64
#     })

# @app.route("/health", methods=["GET"])
# def health():
#     return jsonify({"status":"ok"})

# if __name__ == "__main__":
#     import re # Used for regex in advanced query parsing (humidity threshold)
#     app.run(debug=True, port=3030)


from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import pandas as pd
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import re
import numpy as np

app = Flask(__name__)
CORS(app)

# Cache for bike data
bike_data_cache = {
    'data': None,
    'timestamp': None,
    'cache_duration': 30  # 30 seconds cache
}

def get_dublin_bikes():
    """Get Dublin bikes data with caching"""
    current_time = datetime.now()
    
    # Check if cache is valid
    if (bike_data_cache['data'] is not None and 
        bike_data_cache['timestamp'] is not None and
        (current_time - bike_data_cache['timestamp']).total_seconds() < bike_data_cache['cache_duration']):
        return bike_data_cache['data']
    
    try:
        url = "https://api.cyclocity.fr/contracts/dublin/gbfs/station_status.json"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        # Update cache
        bike_data_cache['data'] = data
        bike_data_cache['timestamp'] = current_time
        
        return data
    except Exception as e:
        # Return cached data even if stale when API fails
        if bike_data_cache['data']:
            return bike_data_cache['data']
        raise e

def generate_chart(df, title, ylabel, xlabel="Station ID", chart_type='bar'):
    """Generate chart from DataFrame"""
    plt.figure(figsize=(10, 6))
    
    if chart_type == 'bar':
        bars = plt.bar(df.index.astype(str), df.values, color='skyblue', alpha=0.7)
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)
    elif chart_type == 'pie':
        plt.pie(df.values, labels=df.index.astype(str), autopct='%1.1f%%', startangle=90)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def analyze_bike_data(stations_df):
    """Perform comprehensive analysis on bike station data"""
    analysis = {}
    
    # Basic statistics
    analysis['total_stations'] = len(stations_df)
    analysis['total_bikes'] = int(stations_df['num_bikes_available'].sum())
    analysis['total_docks'] = int(stations_df['num_docks_available'].sum())
    analysis['total_capacity'] = analysis['total_bikes'] + analysis['total_docks']
    
    # Utilization rate
    analysis['utilization_rate'] = round((analysis['total_bikes'] / analysis['total_capacity'] * 100), 1) if analysis['total_capacity'] > 0 else 0
    
    # Station status counts
    analysis['empty_stations'] = int((stations_df['num_bikes_available'] == 0).sum())
    analysis['full_stations'] = int((stations_df['num_docks_available'] == 0).sum())
    analysis['critical_stations'] = int((stations_df['num_bikes_available'] <= 3).sum())
    analysis['available_stations'] = analysis['total_stations'] - analysis['empty_stations']
    
    # Top and bottom performers
    top_5_bikes = stations_df.nlargest(5, 'num_bikes_available')[['station_id', 'num_bikes_available']]
    bottom_5_bikes = stations_df.nsmallest(5, 'num_bikes_available')[['station_id', 'num_bikes_available']]
    top_5_docks = stations_df.nlargest(5, 'num_docks_available')[['station_id', 'num_docks_available']]
    
    analysis['top_bike_stations'] = [
        {"Station ID": int(row['station_id']), "Available Bikes": int(row['num_bikes_available'])}
        for _, row in top_5_bikes.iterrows()
    ]
    
    analysis['bottom_bike_stations'] = [
        {"Station ID": int(row['station_id']), "Available Bikes": int(row['num_bikes_available'])}
        for _, row in bottom_5_bikes.iterrows()
    ]
    
    analysis['top_dock_stations'] = [
        {"Station ID": int(row['station_id']), "Free Docks": int(row['num_docks_available'])}
        for _, row in top_5_docks.iterrows()
    ]
    
    # Distribution analysis
    analysis['avg_bikes_per_station'] = round(stations_df['num_bikes_available'].mean(), 1)
    analysis['median_bikes'] = int(stations_df['num_bikes_available'].median())
    analysis['std_bikes'] = round(stations_df['num_bikes_available'].std(), 1)
    
    # Calculate availability zones
    analysis['high_availability'] = int((stations_df['num_bikes_available'] > 20).sum())
    analysis['medium_availability'] = int(((stations_df['num_bikes_available'] >= 5) & (stations_df['num_bikes_available'] <= 20)).sum())
    analysis['low_availability'] = int(((stations_df['num_bikes_available'] >= 1) & (stations_df['num_bikes_available'] < 5)).sum())
    
    # Recommendations
    recommendations = []
    
    if analysis['empty_stations'] > 5:
        recommendations.append(f"🚨 Critical: {analysis['empty_stations']} stations are completely empty. Immediate bike redistribution needed!")
    
    if analysis['critical_stations'] > 10:
        recommendations.append(f"⚠️ Warning: {analysis['critical_stations']} stations have 3 or fewer bikes. Consider rebalancing soon.")
    
    if analysis['utilization_rate'] > 80:
        recommendations.append("📈 High system utilization! Consider adding more bikes to meet demand.")
    elif analysis['utilization_rate'] < 30:
        recommendations.append("📉 Low system utilization. You may have excess bikes in the system.")
    
    if len(recommendations) == 0:
        recommendations.append("✅ System is operating well within normal parameters.")
    
    analysis['recommendations'] = recommendations
    
    # Generate insights
    insights = []
    insights.append(f"📊 System Overview: {analysis['total_bikes']} bikes available across {analysis['total_stations']} stations")
    insights.append(f"⚡ Utilization Rate: {analysis['utilization_rate']}%")
    insights.append(f"🔄 Availability: {analysis['available_stations']} stations have bikes, {analysis['empty_stations']} are empty")
    insights.append(f"📈 Top Station: Station {analysis['top_bike_stations'][0]['Station ID']} has {analysis['top_bike_stations'][0]['Available Bikes']} bikes")
    insights.append(f"📉 Critical Stations: {analysis['critical_stations']} stations have 3 or fewer bikes")
    
    analysis['insights'] = insights
    
    return analysis

def process_query_nlp(query):
    """Process natural language query using simple NLP"""
    query_lower = query.lower().strip()
    
    # Check for greeting/introduction
    if any(word in query_lower for word in ['hello', 'hi', 'hey', 'greeting', 'start', 'help']):
        return {
            'type': 'greeting',
            'message': 'Hello! I am your Urban Data Assistant. You can ask me about:\n• Bike station availability\n• Station capacity in Dublin\n• Empty stations\n• Available bikes\n• System status\n• Analysis report\n\nTry asking questions like:\n• "Show me available bikes"\n• "Which stations are empty?"\n• "What is the system status?"\n• "Station capacity in Dublin"\n• "Generate analysis report"\n• "Analyze bike data"'
        }
    
    # Check for analysis
    if any(phrase in query_lower for phrase in ['analyze', 'analysis', 'report', 'summary', 'insights', 'overview']):
        return {'type': 'analysis'}
    
    # Check for available bikes
    if any(phrase in query_lower for phrase in ['available bikes', 'bikes available', 'how many bikes', 'free bikes']):
        return {'type': 'available_bikes', 'top_n': 10}
    
    # Check for empty stations
    if any(phrase in query_lower for phrase in ['empty station', 'no bikes', 'zero bikes', '0 bikes']):
        return {'type': 'empty_stations'}
    
    # Check for station capacity
    if any(phrase in query_lower for phrase in ['station capacity', 'capacity in dublin', 'total capacity', 'bike capacity']):
        return {'type': 'station_capacity'}
    
    # Check for full stations
    if any(phrase in query_lower for phrase in ['full station', 'no docks', 'full capacity']):
        return {'type': 'full_stations'}
    
    # Check for system status
    if any(phrase in query_lower for phrase in ['system status', 'overall status', 'total status', 'system overview']):
        return {'type': 'system_status'}
    
    # Check for specific station
    station_match = re.search(r'station\s+(\d+)', query_lower)
    if station_match:
        return {'type': 'specific_station', 'station_id': int(station_match.group(1))}
    
    # Default response for unrecognized queries
    return {
        'type': 'unknown',
        'message': "I can help you with Dublin bike data! Try asking:\n• 'available bikes'\n• 'empty stations'\n• 'station capacity'\n• 'system status'\n• 'analysis report'\n• 'status of station 90'"
    }

@app.route("/api/query", methods=["POST"])
def query():
    """Main query endpoint for natural language queries"""
    try:
        data = request.get_json()
        user_query = data.get("query", "").strip()
        
        if not user_query:
            return jsonify({
                "message": "Please enter a query about Dublin bikes.",
                "result": [],
                "image": None,
                "analysis": None
            })
        
        # Process the query using NLP
        nlp_result = process_query_nlp(user_query)
        
        # Get bike data
        try:
            bikes_data = get_dublin_bikes()
            stations_df = pd.DataFrame(bikes_data["data"]["stations"])
        except Exception as e:
            return jsonify({
                "message": f"Unable to fetch bike data: {str(e)}",
                "result": [],
                "image": None,
                "analysis": None
            }), 500
        
        response_data = {
            "message": "",
            "result": [],
            "image": None,
            "analysis": None
        }
        
        # Handle different query types
        query_type = nlp_result['type']
        
        if query_type == 'greeting':
            response_data['message'] = nlp_result['message']
            
        elif query_type == 'analysis':
            # Perform comprehensive analysis
            analysis = analyze_bike_data(stations_df)
            
            # Format analysis message
            analysis_text = "📊 **Comprehensive Dublin Bikes Analysis Report**\n\n"
            
            # Add insights
            analysis_text += "**Key Insights:**\n"
            for insight in analysis['insights']:
                analysis_text += f"• {insight}\n"
            
            analysis_text += "\n**Recommendations:**\n"
            for rec in analysis['recommendations']:
                analysis_text += f"• {rec}\n"
            
            analysis_text += f"\n**Station Performance:**\n"
            analysis_text += f"• High availability (>20 bikes): {analysis['high_availability']} stations\n"
            analysis_text += f"• Medium availability (5-20 bikes): {analysis['medium_availability']} stations\n"
            analysis_text += f"• Low availability (1-4 bikes): {analysis['low_availability']} stations\n"
            analysis_text += f"• Empty stations: {analysis['empty_stations']} stations\n"
            
            response_data['message'] = analysis_text
            response_data['analysis'] = analysis
            
            # Generate distribution chart
            dist_data = pd.Series({
                'High (>20)': analysis['high_availability'],
                'Medium (5-20)': analysis['medium_availability'],
                'Low (1-4)': analysis['low_availability'],
                'Empty': analysis['empty_stations']
            })
            response_data['image'] = generate_chart(
                dist_data, 
                "Bike Availability Distribution",
                "Number of Stations",
                "Availability Level",
                chart_type='pie'
            )
            
        elif query_type == 'available_bikes':
            top_n = nlp_result.get('top_n', 10)
            
            # Sort by available bikes
            stations_df = stations_df.sort_values('num_bikes_available', ascending=False).head(top_n)
            
            # Create result list
            result_list = []
            for _, row in stations_df.iterrows():
                result_list.append({
                    "Station ID": int(row['station_id']),
                    "Available Bikes": int(row['num_bikes_available']),
                    "Free Docks": int(row['num_docks_available']),
                    "Total Docks": int(row['num_docks_available'] + row['num_bikes_available'])
                })
            
            response_data['message'] = f"Top {len(result_list)} stations with most available bikes:"
            response_data['result'] = result_list
            
            # Generate chart
            chart_data = stations_df.set_index('station_id')['num_bikes_available']
            response_data['image'] = generate_chart(
                chart_data, 
                f"Top {len(result_list)} Stations by Available Bikes",
                "Available Bikes"
            )
            
        elif query_type == 'empty_stations':
            empty_stations = stations_df[stations_df['num_bikes_available'] == 0]
            
            if len(empty_stations) == 0:
                response_data['message'] = "Great news! There are currently no empty stations in Dublin."
            else:
                result_list = []
                for _, row in empty_stations.iterrows():
                    result_list.append({
                        "Station ID": int(row['station_id']),
                        "Available Bikes": 0,
                        "Free Docks": int(row['num_docks_available']),
                        "Total Docks": int(row['num_docks_available'])
                    })
                
                response_data['message'] = f"There are {len(result_list)} empty stations (0 bikes available):"
                response_data['result'] = result_list
                
                # Generate chart
                chart_data = empty_stations.set_index('station_id')['num_docks_available']
                response_data['image'] = generate_chart(
                    chart_data,
                    "Empty Stations - Free Docks Available",
                    "Free Docks",
                    "Station ID"
                )
        
        elif query_type == 'station_capacity':
            # Calculate capacity statistics
            stations_df['total_capacity'] = stations_df['num_bikes_available'] + stations_df['num_docks_available']
            
            total_bikes = stations_df['num_bikes_available'].sum()
            total_docks = stations_df['num_docks_available'].sum()
            total_capacity = total_bikes + total_docks
            avg_bikes_per_station = stations_df['num_bikes_available'].mean()
            avg_capacity_per_station = stations_df['total_capacity'].mean()
            
            # Find stations with highest capacity
            high_capacity = stations_df.nlargest(5, 'total_capacity')[['station_id', 'total_capacity']]
            
            result_list = []
            for _, row in high_capacity.iterrows():
                result_list.append({
                    "Station ID": int(row['station_id']),
                    "Total Capacity": int(row['total_capacity'])
                })
            
            response_data['message'] = (
                f"Dublin Bike Stations Capacity Overview:\n"
                f"• Total bikes in system: {int(total_bikes)}\n"
                f"• Total free docks: {int(total_docks)}\n"
                f"• Total capacity: {int(total_capacity)}\n"
                f"• Average bikes per station: {avg_bikes_per_station:.1f}\n"
                f"• Average station capacity: {avg_capacity_per_station:.1f}\n\n"
                f"Top {len(result_list)} stations by total capacity:"
            )
            response_data['result'] = result_list
            
            # Generate chart for high capacity stations
            chart_data = high_capacity.set_index('station_id')['total_capacity']
            response_data['image'] = generate_chart(
                chart_data,
                "Top 5 Stations by Total Capacity",
                "Total Capacity (Bikes + Docks)"
            )
        
        elif query_type == 'full_stations':
            full_stations = stations_df[stations_df['num_docks_available'] == 0]
            
            if len(full_stations) == 0:
                response_data['message'] = "No stations are completely full. There are free docks available at all stations."
            else:
                result_list = []
                for _, row in full_stations.iterrows():
                    result_list.append({
                        "Station ID": int(row['station_id']),
                        "Available Bikes": int(row['num_bikes_available']),
                        "Free Docks": 0,
                        "Total Docks": int(row['num_bikes_available'])
                    })
                
                response_data['message'] = f"There are {len(result_list)} completely full stations (0 free docks):"
                response_data['result'] = result_list
        
        elif query_type == 'system_status':
            total_bikes = stations_df['num_bikes_available'].sum()
            total_docks = stations_df['num_docks_available'].sum()
            total_stations = len(stations_df)
            
            # Calculate utilization
            total_capacity = total_bikes + total_docks
            utilization_rate = (total_bikes / total_capacity * 100) if total_capacity > 0 else 0
            
            # Count stations by status
            empty_stations = len(stations_df[stations_df['num_bikes_available'] == 0])
            full_stations = len(stations_df[stations_df['num_docks_available'] == 0])
            active_stations = total_stations - empty_stations
            
            response_data['message'] = (
                f"🚲 DublinBikes System Status 🚲\n"
                f"• Total Stations: {total_stations}\n"
                f"• Active Stations: {active_stations}\n"
                f"• Empty Stations: {empty_stations}\n"
                f"• Full Stations: {full_stations}\n"
                f"• Total Bikes Available: {int(total_bikes)}\n"
                f"• Total Free Docks: {int(total_docks)}\n"
                f"• System Utilization: {utilization_rate:.1f}%\n"
                f"• Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            # Create status summary
            response_data['result'] = [{
                "Metric": "Total Stations",
                "Value": total_stations
            }, {
                "Metric": "Available Bikes",
                "Value": int(total_bikes)
            }, {
                "Metric": "Free Docks",
                "Value": int(total_docks)
            }, {
                "Metric": "Utilization Rate",
                "Value": f"{utilization_rate:.1f}%"
            }]
        
        elif query_type == 'specific_station':
            station_id = nlp_result['station_id']
            station_data = stations_df[stations_df['station_id'] == station_id]
            
            if len(station_data) == 0:
                response_data['message'] = f"Station {station_id} not found. Valid station IDs range from 1 to {stations_df['station_id'].max()}."
            else:
                row = station_data.iloc[0]
                total_docks = row['num_bikes_available'] + row['num_docks_available']
                utilization = (row['num_bikes_available'] / total_docks * 100) if total_docks > 0 else 0
                
                response_data['message'] = (
                    f"Station {station_id} Status:\n"
                    f"• Available Bikes: {int(row['num_bikes_available'])}\n"
                    f"• Free Docks: {int(row['num_docks_available'])}\n"
                    f"• Total Capacity: {int(total_docks)}\n"
                    f"• Utilization: {utilization:.1f}%\n"
                    f"• Renting Enabled: {'Yes' if row['is_renting'] else 'No'}\n"
                    f"• Returning Enabled: {'Yes' if row['is_returning'] else 'No'}"
                )
                
                response_data['result'] = [{
                    "Station ID": int(station_id),
                    "Available Bikes": int(row['num_bikes_available']),
                    "Free Docks": int(row['num_docks_available']),
                    "Total Capacity": int(total_docks),
                    "Utilization": f"{utilization:.1f}%"
                }]
        
        else:  # unknown query type
            response_data['message'] = nlp_result['message']
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            "message": f"Error processing query: {str(e)}",
            "result": [],
            "image": None,
            "analysis": None
        }), 500

@app.route("/api/bikes", methods=["GET"])
def bikes():
    """Direct API endpoint for raw bike data"""
    try:
        data = get_dublin_bikes()
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/analyze", methods=["GET"])
def analyze_endpoint():
    """Direct analysis endpoint"""
    try:
        bikes_data = get_dublin_bikes()
        stations_df = pd.DataFrame(bikes_data["data"]["stations"])
        analysis = analyze_bike_data(stations_df)
        return jsonify(analysis)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Add health endpoint for compatibility
@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for frontend compatibility"""
    return jsonify({"status": "ok"})

@app.route("/api/health", methods=["GET"])
def api_health():
    """Detailed health check endpoint"""
    try:
        get_dublin_bikes()
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "Dublin Bikes API"
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route("/")
def index():
    """Root endpoint with API info"""
    return jsonify({
        "name": "Dublin Bikes Analytics API",
        "version": "1.0.0",
        "endpoints": {
            "POST /api/query": "Process natural language queries about Dublin bikes",
            "GET /api/bikes": "Get raw Dublin bikes data",
            "GET /api/analyze": "Get comprehensive analysis report",
            "GET /health": "Simple health check",
            "GET /api/health": "Detailed health check"
        },
        "example_queries": [
            "available bikes",
            "empty stations",
            "station capacity in Dublin",
            "system status",
            "analysis report",
            "status of station 90"
        ]
    })

if __name__ == "__main__":
    print("Starting Dublin Bikes Analytics API...")
    print("Available endpoints:")
    print("  POST /api/query - Process natural language queries")
    print("  GET  /api/bikes - Get raw bike data")
    print("  GET  /api/analyze - Get analysis report")
    print("  GET  /health - Simple health check")
    print("  GET  /api/health - Detailed health check")
    print("\nExample queries: 'available bikes', 'empty stations', 'station capacity', 'system status', 'analysis report'")
    app.run(host="0.0.0.0", port=3030, debug=True)