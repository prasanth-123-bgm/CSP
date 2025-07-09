import requests

weatherapi_key = "99aef283258d437fac031135250807"  # replace with your real key

def get_weather(city):
    url = f"http://api.weatherapi.com/v1/current.json?key={weatherapi_key}&q={city}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        condition = data['current']['condition']['text']
        temp = data['current']['temp_c']
        humidity = data['current']['humidity']
        wind = data['current']['wind_kph']
        report = (
            f"Weather in {city}:\n"
            f"Condition: {condition}\n"
            f"Temperature: {temp}°C\n"
            f"Humidity: {humidity}%\n"
            f"Wind Speed: {wind} kph"
        )
        return report
    else:
        return f"❌ Weather information not available for {city}."

def get_weather_telugu(city):
    url = f"http://api.weatherapi.com/v1/current.json?key={weatherapi_key}&q={city}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        condition = data['current']['condition']['text']
        temp = data['current']['temp_c']
        humidity = data['current']['humidity']
        wind = data['current']['wind_kph']
        report = (
           f"{city} లో వాతావరణ నివేదిక:\n"
            f"పరిస్థితి: {condition}\n"
            f"ఉష్ణోగ్రత: {temp} డిగ్రీల సెల్సియస్\n"
            f"తేమ శాతం: {humidity}%\n"
            f"గాలివేగం: {wind} కి.మీ/గంటకి"
        )
        return report
    else:
       return f"❌ {city} యొక్క వాతావరణాన్ని పొందలేకపోయాము."


def get_weather_hindi(city):
    url = f"http://api.weatherapi.com/v1/current.json?key={weatherapi_key}&q={city}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        condition = data['current']['condition']['text']
        temp = data['current']['temp_c']
        humidity = data['current']['humidity']
        wind = data['current']['wind_kph']
        report = (
            f"{city} में मौसम की जानकारी:\n"
            f"स्थिति: {condition}\n"
            f"तापमान: {temp} डिग्री सेल्सियस\n"
            f"नमी: {humidity}%\n"
            f"हवा की गति: {wind} किलोमीटर प्रति घंटा"
)

        return report
    else:
       return f"❌ {city} के मौसम की जानकारी प्राप्त नहीं हो सकी।"
