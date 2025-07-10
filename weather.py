import requests
import os

# Use your WeatherAPI key here or from env
WEATHERAPI_KEY = os.getenv("WEATHERAPI_KEY", "99aef283258d437fac031135250807")

# 1️⃣ Get Latitude & Longitude from Nominatim for any location
def get_coordinates(place_name):
    try:
        url = f"https://nominatim.openstreetmap.org/search"
        params = {
            'q': place_name,
            'format': 'json',
            'limit': 1
        }
        headers = {
            'User-Agent': 'AgriVoice-Pro/1.0 (contact@example.com)'  # Customize if needed
        }
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        results = response.json()
        if results:
            lat = results[0]["lat"]
            lon = results[0]["lon"]
            return lat, lon
        else:
            return None, None
    except Exception as e:
        return None, None

# 2️⃣ Use WeatherAPI with Coordinates
def fetch_weather(city_name, lang="en"):
    lat, lon = get_coordinates(city_name)
    if not lat or not lon:
        return f"❌ Could not locate the place '{city_name}'. Please try again with more details."

    url = f"http://api.weatherapi.com/v1/current.json"
    params = {
        "key": WEATHERAPI_KEY,
        "q": f"{lat},{lon}",
        "lang": lang
    }

    try:
        res = requests.get(url, params=params, timeout=5)
        res.raise_for_status()
        data = res.json()["current"]
        condition = data['condition']['text']
        temp = data['temp_c']
        humidity = data['humidity']
        wind = data['wind_kph']
    except Exception as e:
        return "❌ Failed to retrieve weather data."

    if lang == "te":
        return (
            f"{city_name} లో వాతావరణ నివేదిక:\n"
            f"పరిస్థితి: {condition}\n"
            f"ఉష్ణోగ్రత: {temp}°C\n"
            f"తేమ: {humidity}%\n"
            f"గాలి వేగం: {wind} కిమీ/గం"
        )
    elif lang == "hi":
        return (
            f"{city_name} में मौसम जानकारी:\n"
            f"स्थिति: {condition}\n"
            f"तापमान: {temp}°C\n"
            f"नमी: {humidity}%\n"
            f"हवा की गति: {wind} किमी/घंटा"
        )
    else:
        return (
            f"Weather in {city_name}:\n"
            f"Condition: {condition}\n"
            f"Temperature: {temp}°C\n"
            f"Humidity: {humidity}%\n"
            f"Wind Speed: {wind} kph"
        )

# 3️⃣ Language-specific wrappers
def get_weather(city):
    return fetch_weather(city, lang="en")

def get_weather_telugu(city):
    return fetch_weather(city, lang="te")

def get_weather_hindi(city):
    return fetch_weather(city, lang="hi")
