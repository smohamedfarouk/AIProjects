package com.jupiter.langchaincommands.search;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.Map;

/**
 * Minimal Tavily client - the Java version of {@code TavilyClient(api_key=...).search(query=..., max_results=...)}.
 * <p>
 * Tavily is a web-search API made for LLM apps: instead of raw web pages it returns short, clean text snippets with
 * their titles and URLs, ready to put in front of a model. {@link DeepAgent} exposes it to the model as the
 * {@code internet_search} tool.
 */
public final class TavilySearch {

    private static final URI SEARCH_URL = URI.create("https://api.tavily.com/search");
    private static final ObjectMapper JSON = new ObjectMapper();

    private final String apiKey;
    private final HttpClient http = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(20)).build();

    /**
     * @param apiKey Tavily API key (TAVILY_API_KEY); a missing key fails here, before any agent work starts
     */
    public TavilySearch(String apiKey) {
        if (apiKey == null || apiKey.isBlank()) {
            throw new IllegalStateException("TAVILY_API_KEY not found in environment or .env file.");
        }
        this.apiKey = apiKey;
    }

    /**
     * Runs one web search.
     * <p>
     * Sends {@code {"query": ..., "max_results": ...}} to {@code https://api.tavily.com/search} with the API key as a
     * Bearer token, and returns Tavily's JSON reply unchanged (the query, and results with title, url, content and a
     * relevance score). The JSON goes straight back to the model, which reads it and decides what to do next.
     *
     * @param query      what to search for, e.g. "AI news September 16 2026"
     * @param maxResults how many results to return (the agent normally asks for 5)
     * @return Tavily's JSON response text
     * @throws RuntimeException on a network error or a non-200 reply (e.g. an invalid key), with Tavily's message
     */
    public String search(String query, int maxResults) {
        try {
            String body = JSON.writeValueAsString(Map.of("query", query, "max_results", maxResults));
            HttpRequest request = HttpRequest.newBuilder(SEARCH_URL)
                    .timeout(Duration.ofSeconds(60))
                    .header("Authorization", "Bearer " + apiKey)
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(body))
                    .build();
            HttpResponse<String> response = http.send(request, HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() != 200) {
                throw new IOException("Tavily returned HTTP " + response.statusCode() + ": " + response.body());
            }
            return response.body();
        } catch (IOException e) {
            throw new RuntimeException("Tavily search failed: " + e.getMessage(), e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Tavily search interrupted", e);
        }
    }
}
