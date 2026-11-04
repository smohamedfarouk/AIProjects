package com.bepec.deepagents.tools;

import com.fasterxml.jackson.databind.ObjectMapper;
import dev.langchain4j.agent.tool.P;
import dev.langchain4j.agent.tool.Tool;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.Map;

/**
 * Python:
 * <pre>
 * tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
 * def internet_search(query: str, max_results: int = 5):
 *     return tavily.search(query=query, max_results=max_results)
 * </pre>
 * Calls the Tavily REST API directly and returns the raw JSON response to the model.
 */
public class InternetSearchTool {

    private static final URI TAVILY_SEARCH = URI.create("https://api.tavily.com/search");
    private static final ObjectMapper JSON = new ObjectMapper();

    private final String apiKey;
    private final HttpClient http = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(20)).build();

    public InternetSearchTool(String apiKey) {
        if (apiKey == null || apiKey.isBlank()) {
            throw new IllegalArgumentException("TAVILY_API_KEY is not set");
        }
        this.apiKey = apiKey;
    }

    @Tool(name = "internet_search", value = """
            Search the web for current information on topic. Use for any question \
            that needs up-to-date facts, trends or competitor/market information.""")
    public String internetSearch(@P("the search query") String query,
                                 @P(value = "maximum number of results (default 5)", required = false) Integer maxResults) {
        int max = maxResults == null || maxResults <= 0 ? 5 : maxResults;
        try {
            String body = JSON.writeValueAsString(Map.of("query", query, "max_results", max));
            HttpRequest request = HttpRequest.newBuilder(TAVILY_SEARCH)
                    .timeout(Duration.ofSeconds(60))
                    .header("Content-Type", "application/json")
                    .header("Authorization", "Bearer " + apiKey)
                    .POST(HttpRequest.BodyPublishers.ofString(body))
                    .build();
            HttpResponse<String> response = http.send(request, HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() / 100 != 2) {
                return "Search failed (HTTP " + response.statusCode() + "): " + response.body();
            }
            return response.body();
        } catch (IOException e) {
            return "Search failed: " + e.getMessage();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return "Search interrupted";
        }
    }
}
