import requests
import os

# 🔑 Use your API key securely (remember not to expose it in code)
WEATHERAPI_KEY = os.getenv("WEATHERAPI_KEY", "99aef283258d437fac031135250807")

def fetch_weather(city=None, lat=None, lon=None, lang=None):
    """
    Fetches current weather using WeatherAPI.com.
    Accepts either city name _or_ latitude & longitude.
    Supports localization (lang code like 'te'/'hi').
    """
    if not WEATHERAPI_KEY or (not city and not (lat and lon)):
        return "Invalid input or missing API key."

    # Build URL
    base = "http://api.weatherapi.com/v1/current.json"
    q = f"{lat},{lon}" if lat and lon else city
    params = {
        "key": WEATHERAPI_KEY,
        "q": q,
        "lang": lang if lang in ["en", "te", "hi"] else "en"
    }

    try:
        res = requests.get(base, params=params, timeout=5)
        res.raise_for_status()
        data = res.json()["current"]
    except requests.RequestException:
        return f"❌ Could not fetch weather for {q}."
    except KeyError:
        return "❌ Unexpected response structure from API."

    # Parse values
    condition = data.get("condition", {}).get("text", "Unknown")
    temp = data.get("temp_c", "N/A")
    humidity = data.get("humidity", "N/A")
    wind = data.get("wind_kph", "N/A")

    # Localization templates
    templates = {
        "en": (
            f"Weather in {city or f'{lat},{lon}'}:\n"
            f"Condition: {condition}\n"
            f"Temperature: {temp}°C\n"
            f"Humidity: {humidity}%\n"
            f"Wind Speed: {wind} kph"
        ),
        "te": (
            f"{city} లో వాతావరణ నివేదిక:\n"
            f"పరిస్థితి: {condition}\n"
            f"ఉష్ణోగ్రత: {temp}°C\n"
            f"తేమ: {humidity}%\n"
            f"గాలి వేగం: {wind} కిమీ/గం"
        ),
        "hi": (
            f"{city} में मौसम जानकारी:\n"
            f"स्थिति: {condition}\n"
            f"तापमान: {temp}°C\n"
            f"नमी: {humidity}%\n"
            f"हवा की गति: {wind} किमी/घंटा"
        )
    }

    return templates.get(lang, templates["en"])

# Aliased functions
def get_weather(city):
    return fetch_weather(city=city, lang="en")

def get_weather_telugu(city):
    return fetch_weather(city=city, lang="te")

def get_weather_hindi(city):
    return fetch_weather(city=city, lang="hi")
