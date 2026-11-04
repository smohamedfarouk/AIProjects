package com.bepec.mcp;

import com.bepec.mcp.McpTools.Param;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.modelcontextprotocol.server.McpServerFeatures.SyncToolSpecification;

import java.io.IOException;
import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.StringJoiner;

/**
 * Java port of weather_server.py - an MCP server named "WeatherTools" backed by the free Open-Meteo API
 * (no API key needed).
 * <ul>
 *   <li>get_current_weather(city) - condition, temperature, feels-like, humidity, wind</li>
 *   <li>get_forecast(city, days = 3) - daily forecast for 1 to 7 days</li>
 * </ul>
 * Not connected by default: {@link AgentMain} only starts {@link MathServerMain}, exactly like agent.py.
 * See {@link AgentMain} for the one line that connects it.
 */
public final class WeatherServerMain {

    static final String GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search";
    static final String FORECAST_URL = "https://api.open-meteo.com/v1/forecast";

    static final Map<Integer, String> WEATHER_CODES = Map.ofEntries(
            Map.entry(0, "Clear sky"), Map.entry(1, "Mainly clear"), Map.entry(2, "Partly cloudy"),
            Map.entry(3, "Overcast"), Map.entry(45, "Fog"), Map.entry(48, "Rime fog"),
            Map.entry(51, "Light drizzle"), Map.entry(53, "Drizzle"), Map.entry(55, "Heavy drizzle"),
            Map.entry(61, "Light rain"), Map.entry(63, "Rain"), Map.entry(65, "Heavy rain"),
            Map.entry(71, "Light snow"), Map.entry(73, "Snow"), Map.entry(75, "Heavy snow"),
            Map.entry(80, "Light showers"), Map.entry(81, "Showers"), Map.entry(82, "Violent showers"),
            Map.entry(95, "Thunderstorm"), Map.entry(96, "Thunderstorm with hail"),
            Map.entry(99, "Severe thunderstorm with hail"));

    private static final HttpClient HTTP = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(15)).build();
    private static final ObjectMapper JSON = new ObjectMapper();

    private WeatherServerMain() {
    }

    /** Turns an Open-Meteo weather code (WMO code) into words, e.g. 3 -> "Overcast"; unknown codes -> "Unknown". */
    static String describe(int code) {
        return WEATHER_CODES.getOrDefault(code, "Unknown");
    }

    /** Keeps the forecast length between 1 and 7 days - the range the tool promises the model. */
    static int clampDays(int days) {
        return Math.max(1, Math.min(days, 7));
    }

    /** Looks up a city name and returns its first match (name, country, latitude, longitude), or null. */
    static JsonNode findCity(String city) {
        JsonNode results = get(GEOCODE_URL + "?count=1&name=" + encode(city)).path("results");
        return results.isArray() && !results.isEmpty() ? results.get(0) : null;
    }

    /**
     * The get_current_weather tool.
     * <ol>
     *   <li>Find the city with {@link #findCity} - if there is no match, return a friendly message instead of
     *       failing, so the model can tell the user.</li>
     *   <li>Ask the forecast API for the "current" values at the city's latitude/longitude (timezone=auto gives
     *       the local time of the city).</li>
     *   <li>Format condition, temperature, feels-like temperature, humidity and wind as readable lines.</li>
     * </ol>
     *
     * @param city city name typed by the user, e.g. "London"
     * @return a short multi-line weather report, or a "Could not find" message
     */
    static String currentWeather(String city) {
        JsonNode place = findCity(city);
        if (place == null) {
            return "Could not find a city called '" + city + "'.";
        }
        JsonNode now = get(FORECAST_URL + coordinates(place)
                + "&current=temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m"
                + "&timezone=auto").path("current");

        return "Weather in " + placeName(place) + " (at " + now.path("time").asText() + "):\n"
                + "- Condition: " + describe(now.path("weather_code").asInt(-1)) + "\n"
                + "- Temperature: " + now.path("temperature_2m").asText() + " °C (feels like "
                + now.path("apparent_temperature").asText() + " °C)\n"
                + "- Humidity: " + now.path("relative_humidity_2m").asText() + "%\n"
                + "- Wind: " + now.path("wind_speed_10m").asText() + " km/h";
    }

    /**
     * The get_forecast tool.
     * <ol>
     *   <li>Limit the number of days to 1-7 ({@link #clampDays}).</li>
     *   <li>Find the city with {@link #findCity}.</li>
     *   <li>Ask the forecast API for daily values: weather code, min/max temperature and rain probability.</li>
     *   <li>Return one line per day, e.g. "- 2026-11-01: Light rain, 6.1-11.4 °C, rain chance 70%".</li>
     * </ol>
     *
     * @param city          city name, e.g. "London"
     * @param requestedDays days asked for by the model (default 3 when it gives none)
     * @return the forecast lines, or a "Could not find" message
     */
    static String forecast(String city, int requestedDays) {
        int days = clampDays(requestedDays);
        JsonNode place = findCity(city);
        if (place == null) {
            return "Could not find a city called '" + city + "'.";
        }
        JsonNode d = get(FORECAST_URL + coordinates(place)
                + "&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max"
                + "&forecast_days=" + days + "&timezone=auto").path("daily");

        StringJoiner lines = new StringJoiner("\n");
        lines.add(days + "-day forecast for " + placeName(place) + ":");
        for (int i = 0; i < d.path("time").size(); i++) {
            lines.add("- " + d.path("time").get(i).asText() + ": "
                    + describe(d.path("weather_code").get(i).asInt(-1)) + ", "
                    + d.path("temperature_2m_min").get(i).asText() + "-"
                    + d.path("temperature_2m_max").get(i).asText() + " °C, "
                    + "rain chance " + d.path("precipitation_probability_max").get(i).asText() + "%");
        }
        return lines.toString();
    }

    /**
     * Describes the two weather tools for MCP (the Java version of {@code @mcp.tool()}): name, description the
     * model reads, JSON schema of the arguments ("city" required, "days" optional), and the handler that calls
     * {@link #currentWeather} or {@link #forecast}.
     *
     * @return the tool definitions served by {@link #main}
     */
    static List<SyncToolSpecification> tools() {
        Param city = Param.required("city", "string", "City name, e.g. London");
        return List.of(
                McpTools.tool("get_current_weather", "Get the current weather for a city.", List.of(city),
                        args -> currentWeather(McpTools.text(args, "city"))),
                McpTools.tool("get_forecast", "Get the daily weather forecast for a city (1 to 7 days, default 3).",
                        List.of(city, Param.optional("days", "integer", "Number of days, 1 to 7")),
                        args -> forecast(McpTools.text(args, "city"),
                                args.containsKey("days") ? (int) McpTools.number(args, "days") : 3)));
    }

    private static String coordinates(JsonNode place) {
        return "?latitude=" + place.path("latitude").asText() + "&longitude=" + place.path("longitude").asText();
    }

    private static String placeName(JsonNode place) {
        return place.path("name").asText() + ", " + place.path("country").asText("");
    }

    private static String encode(String s) {
        return URLEncoder.encode(s, StandardCharsets.UTF_8);
    }

    /**
     * Sends one HTTP GET request to Open-Meteo and parses the JSON reply. Any network problem or non-2xx status
     * becomes an exception, which {@link McpTools#tool} turns into an error result the model can read.
     */
    private static JsonNode get(String url) {
        try {
            HttpRequest request = HttpRequest.newBuilder(URI.create(url)).timeout(Duration.ofSeconds(15)).GET().build();
            HttpResponse<String> response = HTTP.send(request, HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() / 100 != 2) {
                throw new IllegalStateException("Open-Meteo returned HTTP " + response.statusCode());
            }
            return JSON.readTree(response.body());
        } catch (IOException e) {
            throw new IllegalStateException("Open-Meteo request failed: " + e.getMessage(), e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Interrupted", e);
        }
    }

    /**
     * Entry point of the weather MCP server process - the Java version of {@code mcp.run(transport="stdio")}.
     * <p>
     * Started as a child process by {@link MultiServerMcpClient} once you add it to {@link AgentMain#SERVERS}. It then
     * answers {@code initialize} (name "WeatherTools"), {@code tools/list} (get_current_weather, get_forecast) and
     * {@code tools/call} over stdin/stdout. For each call it looks up the city with the Open-Meteo geocoding API,
     * fetches the weather for its latitude/longitude, and returns a readable text summary. When the agent closes
     * this process's stdin, the process exits.
     * <p>
     * Because stdout carries the protocol, this class must never print anything with System.out.
     *
     * @param args not used
     */
    public static void main(String[] args) throws InterruptedException {
        McpTools.serveStdio("WeatherTools", tools());
    }
}
