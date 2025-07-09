import requests
import json

# You'll need to get a free API key from OpenWeatherMap
# Visit: https://openweathermap.org/api
API_KEY = "99aef283258d437fac031135250807"  # Replace with your actual API key
BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

def get_weather(city):
    """Get weather information in English"""
    try:
        if API_KEY == "99aef283258d437fac031135250807":
            return f"Weather service unavailable. Please add your OpenWeatherMap API key in weather.py file."
        
        params = {
            'q': city,
            'appid': API_KEY,
            'units': 'metric'
        }
        
        response = requests.get(BASE_URL, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            temperature = data['main']['temp']
            humidity = data['main']['humidity']
            description = data['weather'][0]['description']
            feels_like = data['main']['feels_like']
            
            weather_report = f"""
            Weather in {city}:
            Temperature: {temperature}°C (feels like {feels_like}°C)
            Humidity: {humidity}%
            Condition: {description.title()}
            """
            
            return weather_report
        else:
            return f"Sorry, couldn't get weather information for {city}. Please check the city name."
    
    except Exception as e:
        return f"Error fetching weather data: {str(e)}"

def get_weather_telugu(city):
    """Get weather information in Telugu"""
    try:
        if API_KEY == "99aef283258d437fac031135250807":
            return f"వాతావరణ సేవ అందుబాటులో లేదు. దయచేసి weather.py ఫైల్‌లో మీ OpenWeatherMap API కీని జోడించండి."
        
        params = {
            'q': city,
            'appid': API_KEY,
            'units': 'metric'
        }
        
        response = requests.get(BASE_URL, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            temperature = data['main']['temp']
            humidity = data['main']['humidity']
            description = data['weather'][0]['description']
            feels_like = data['main']['feels_like']
            
            weather_report = f"""
            {city} లో వాతావరణం:
            ఉష్ణోగ్రత: {temperature}°C (అనుభవం {feels_like}°C)
            తేమ: {humidity}%
            స్థితి: {description}
            """
            
            return weather_report
        else:
            return f"క్షమించండి, {city} కోసం వాతావరణ సమాచారం పొందలేకపోయింది. దయచేసి నగర పేరును తనిఖీ చేయండి."
    
    except Exception as e:
        return f"వాతావరణ డేటా పొందడంలో లోపం: {str(e)}"

def get_weather_hindi(city):
    """Get weather information in Hindi"""
    try:
        if API_KEY == "99aef283258d437fac031135250807":
            return f"मौसम सेवा अनुपलब्ध है। कृपया weather.py फ़ाइल में अपनी OpenWeatherMap API key जोड़ें।"
        
        params = {
            'q': city,
            'appid': API_KEY,
            'units': 'metric'
        }
        
        response = requests.get(BASE_URL, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            temperature = data['main']['temp']
            humidity = data['main']['humidity']
            description = data['weather'][0]['description']
            feels_like = data['main']['feels_like']
            
            weather_report = f"""
            {city} में मौसम:
            तापमान: {temperature}°C (महसूस होता है {feels_like}°C)
            आर्द्रता: {humidity}%
            स्थिति: {description}
            """
            
            return weather_report
        else:
            return f"खुशी है, {city} के लिए मौसम की जानकारी नहीं मिली। कृपया शहर का नाम जांचें।"
    
    except Exception as e:
        return f"मौसम डेटा प्राप्त करने में त्रुटि: {str(e)}"
