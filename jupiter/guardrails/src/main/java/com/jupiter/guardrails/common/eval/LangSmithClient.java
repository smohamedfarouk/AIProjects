package com.jupiter.guardrails.common.eval;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.NullNode;
import com.jupiter.guardrails.common.Config;

import java.io.IOException;
import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.Instant;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;

/**
 * Minimal LangSmith REST client covering what the Python langsmith.Client is used for here:
 * datasets, examples, experiments, runs and feedback.
 */
public class LangSmithClient {

    static final String BASE_URL = "https://api.smith.langchain.com";
    static final int PAGE_SIZE = 100;

    private static final DateTimeFormatter DOTTED_ORDER_TIME =
            DateTimeFormatter.ofPattern("yyyyMMdd'T'HHmmssSSSSSS'Z'").withZone(ZoneOffset.UTC);

    /** A dataset example; id is null for examples not yet uploaded. */
    public record Example(String id, Map<String, Object> inputs, Map<String, Object> outputs) {
        public static Example of(Map<String, Object> inputs, Map<String, Object> outputs) {
            return new Example(null, inputs, outputs);
        }
    }

    private final HttpClient http = HttpClient.newHttpClient();
    private final ObjectMapper json = new ObjectMapper();
    private final String apiKey;

    public LangSmithClient(String apiKey) {
        this.apiKey = apiKey;
    }

    public static LangSmithClient fromConfig() {
        if (Config.LANGCHAIN_API_KEY == null || Config.LANGCHAIN_API_KEY.isBlank()) {
            throw new IllegalStateException("LANGCHAIN_API_KEY not set — cannot run LangSmith evals.");
        }
        return new LangSmithClient(Config.LANGCHAIN_API_KEY);
    }

    public Optional<String> findDatasetId(String name) {
        for (JsonNode ds : request("GET", "/api/v1/datasets?name=" + encode(name), null)) {
            if (name.equals(ds.path("name").asText())) {
                return Optional.of(ds.path("id").asText());
            }
        }
        return Optional.empty();
    }

    /** Creates the dataset with the given examples unless it already exists (like client.has_dataset + create). */
    public String ensureDataset(String name, List<Example> examples) {
        Optional<String> existing = findDatasetId(name);
        if (existing.isPresent()) {
            return existing.get();
        }
        String id = request("POST", "/api/v1/datasets", Map.of("name", name, "data_type", "kv")).path("id").asText();
        for (Example e : examples) {
            request("POST", "/api/v1/examples", Map.of("dataset_id", id, "inputs", e.inputs(), "outputs", e.outputs()));
        }
        return id;
    }

    @SuppressWarnings("unchecked")
    public List<Example> listExamples(String datasetId) {
        List<Example> examples = new ArrayList<>();
        for (int offset = 0; ; offset += PAGE_SIZE) {
            JsonNode page = request("GET", "/api/v1/examples?dataset=" + datasetId
                    + "&offset=" + offset + "&limit=" + PAGE_SIZE, null);
            for (JsonNode e : page) {
                examples.add(new Example(e.path("id").asText(),
                        json.convertValue(e.path("inputs"), Map.class),
                        json.convertValue(e.path("outputs"), Map.class)));
            }
            if (page.size() < PAGE_SIZE) {
                return examples;
            }
        }
    }

    /** An experiment is a tracer session linked to the dataset. Returns its id. */
    public String createExperiment(String name, String datasetId) {
        return request("POST", "/api/v1/sessions", Map.of(
                "name", name,
                "reference_dataset_id", datasetId,
                "start_time", Instant.now().toString())).path("id").asText();
    }

    public void endExperiment(String experimentId) {
        request("PATCH", "/api/v1/sessions/" + experimentId, Map.of("end_time", Instant.now().toString()));
    }

    /** Logs one root run into the experiment, linked to the dataset example it was produced from. */
    public void createRun(UUID runId, String name, String experimentId, String exampleId,
                          Map<String, Object> inputs, Map<String, Object> outputs, String error,
                          Instant start, Instant end) {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("id", runId.toString());
        body.put("trace_id", runId.toString());
        body.put("dotted_order", DOTTED_ORDER_TIME.format(start) + runId);
        body.put("name", name);
        body.put("run_type", "chain");
        body.put("inputs", inputs);
        body.put("outputs", outputs);
        body.put("error", error);
        body.put("start_time", start.toString());
        body.put("end_time", end.toString());
        body.put("session_id", experimentId);
        body.put("reference_example_id", exampleId);
        request("POST", "/api/v1/runs", body);
    }

    public void createFeedback(UUID runId, String key, Double score, String comment) {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("run_id", runId.toString());
        body.put("key", key);
        body.put("score", score);
        body.put("comment", comment);
        request("POST", "/api/v1/feedback", body);
    }

    private JsonNode request(String method, String path, Object body) {
        try {
            HttpRequest.Builder builder = HttpRequest.newBuilder(URI.create(BASE_URL + path))
                    .header("x-api-key", apiKey)
                    .timeout(Duration.ofSeconds(30));
            if (body != null) {
                builder.header("Content-Type", "application/json")
                        .method(method, HttpRequest.BodyPublishers.ofString(json.writeValueAsString(body)));
            } else {
                builder.method(method, HttpRequest.BodyPublishers.noBody());
            }
            HttpRequest req = builder.build();

            for (int attempt = 1; ; attempt++) {
                HttpResponse<String> resp = http.send(req, HttpResponse.BodyHandlers.ofString());
                int status = resp.statusCode();
                if (status / 100 == 2) {
                    return resp.body().isBlank() ? NullNode.getInstance() : json.readTree(resp.body());
                }
                // retry rate limits and server errors
                if ((status == 429 || status >= 500) && attempt < 3) {
                    Thread.sleep(1000L * attempt);
                    continue;
                }
                throw new IllegalStateException("LangSmith " + method + " " + path + " -> " + status + ": " + resp.body());
            }
        } catch (IOException e) {
            throw new IllegalStateException("LangSmith " + method + " " + path + " failed", e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Interrupted calling LangSmith", e);
        }
    }

    private static String encode(String s) {
        return URLEncoder.encode(s, StandardCharsets.UTF_8);
    }
}
