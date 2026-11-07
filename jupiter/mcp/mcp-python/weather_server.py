
import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("WeatherTools")

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Rime fog",
    51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle",
    61: "Light rain", 63: "Rain", 65: "Heavy rain",
    71: "Light snow", 73: "Snow", 75: "Heavy snow",
    80: "Light showers", 81: "Showers", 82: "Violent showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Severe thunderstorm with hail",
}


async def _find_city(city: str) -> dict | None:
   
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(GEOCODE_URL, params={"name": city, "count": 1})
        r.raise_for_status()
        results = r.json().get("results")
        return results[0] if results else None


@mcp.tool()
async def get_current_weather(city: str) -> str:
   
    place = await _find_city(city)
    if not place:
        return f"Could not find a city called '{city}'."

    params = {
        "latitude": place["latitude"],
        "longitude": place["longitude"],
        "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m",
        "timezone": "auto",
    }
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(FORECAST_URL, params=params)
        r.raise_for_status()
        now = r.json()["current"]

    return (
        f"Weather in {place['name']}, {place.get('country', '')} (at {now['time']}):\n"
        f"- Condition: {WEATHER_CODES.get(now['weather_code'], 'Unknown')}\n"
        f"- Temperature: {now['temperature_2m']} °C (feels like {now['apparent_temperature']} °C)\n"
        f"- Humidity: {now['relative_humidity_2m']}%\n"
        f"- Wind: {now['wind_speed_10m']} km/h"
    )


@mcp.tool()
async def get_forecast(city: str, days: int = 3) -> str:
   
    days = max(1, min(days, 7))
    place = await _find_city(city)
    if not place:
        return f"Could not find a city called '{city}'."

    params = {
        "latitude": place["latitude"],
        "longitude": place["longitude"],
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        "forecast_days": days,
        "timezone": "auto",
    }
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(FORECAST_URL, params=params)
        r.raise_for_status()
        d = r.json()["daily"]

    lines = [f"{days}-day forecast for {place['name']}, {place.get('country', '')}:"]
    for i, date in enumerate(d["time"]):
        lines.append(
            f"- {date}: {WEATHER_CODES.get(d['weather_code'][i], 'Unknown')}, "
            f"{d['temperature_2m_min'][i]}-{d['temperature_2m_max'][i]} °C, "
            f"rain chance {d['precipitation_probability_max'][i]}%"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run(transport="stdio")
