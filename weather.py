import requests

# ✅ Set your WeatherAPI.com API key directly here
weatherapi_key = "99aef283258d437fac031135250807"

# 🟢 COMMON: Get weather by city/village name or coordinates
def fetch_weather_data(location):
    """Supports city name, PIN code, or coordinates (lat,lon) as string."""
    url = f"http://api.weatherapi.com/v1/current.json?key={weatherapi_key}&q={location}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

# 🌐 English Weather
def get_weather(location):
    data = fetch_weather_data(location)
    if data:
        city = data['location']['name']
        region = data['location']['region']
        country = data['location']['country']
        condition = data['current']['condition']['text']
        temp = data['current']['temp_c']
        humidity = data['current']['humidity']
        wind = data['current']['wind_kph']
        return (
            f"Weather in {city}, {region}, {country}:\n"
            f"Condition: {condition}\n"
            f"Temperature: {temp}°C\n"
            f"Humidity: {humidity}%\n"
            f"Wind Speed: {wind} kph"
        )
    else:
        return f"❌ Weather information not available for '{location}'."

# 🌐 Telugu Weather
def get_weather_telugu(location):
    data = fetch_weather_data(location)
    if data:
        city = data['location']['name']
        condition = data['current']['condition']['text']
        temp = data['current']['temp_c']
        humidity = data['current']['humidity']
        wind = data['current']['wind_kph']
        return (
            f"{city} లో వాతావరణ నివేదిక:\n"
            f"పరిస్థితి: {condition}\n"
            f"ఉష్ణోగ్రత: {temp}°C\n"
            f"తేమ: {humidity}%\n"
            f"గాలివేగం: {wind} కి.మీ/గం"
        )
    else:
        return f"❌ '{location}' యొక్క వాతావరణాన్ని పొందలేకపోయాము."

# 🌐 Hindi Weather
def get_weather_hindi(location):
    data = fetch_weather_data(location)
    if data:
        city = data['location']['name']
        condition = data['current']['condition']['text']
        temp = data['current']['temp_c']
        humidity = data['current']['humidity']
        wind = data['current']['wind_kph']
        return (
            f"{city} में मौसम की जानकारी:\n"
            f"स्थिति: {condition}\n"
            f"तापमान: {temp}°C\n"
            f"नमी: {humidity}%\n"
            f"हवा की गति: {wind} किलोमीटर/घंटा"
        )
    else:
        return f"❌ '{location}' के मौसम की जानकारी प्राप्त नहीं हो सकी।"
