package com.jupiter.guardrails.common;

import io.opentelemetry.api.OpenTelemetry;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.api.common.Attributes;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.StatusCode;
import io.opentelemetry.api.trace.Tracer;
import io.opentelemetry.context.Scope;
import io.opentelemetry.exporter.otlp.http.trace.OtlpHttpSpanExporter;
import io.opentelemetry.sdk.OpenTelemetrySdk;
import io.opentelemetry.sdk.resources.Resource;
import io.opentelemetry.sdk.trace.SdkTracerProvider;
import io.opentelemetry.sdk.trace.export.BatchSpanProcessor;

import java.util.function.Consumer;
import java.util.function.Function;

/** OpenTelemetry tracing exported to LangSmith (Java equivalent of @traceable). No-op when tracing is disabled. */
public final class Tracing {

    static final String LANGSMITH_OTEL_ENDPOINT = "https://api.smith.langchain.com/otel/v1/traces";

    private static OpenTelemetry otel = OpenTelemetry.noop();
    private static SdkTracerProvider provider;
    private static Tracer tracer = otel.getTracer("guardrails");

    private Tracing() {
    }

    public static void init() {
        init(Config.LANGCHAIN_PROJECT);
    }

    /** Starts exporting spans to the given LangSmith project. */
    public static void init(String project) {
        if (!Config.LANGCHAIN_TRACING_V2 || Config.LANGCHAIN_API_KEY == null || Config.LANGCHAIN_API_KEY.isBlank()) {
            return;
        }
        OtlpHttpSpanExporter exporter = OtlpHttpSpanExporter.builder()
                .setEndpoint(LANGSMITH_OTEL_ENDPOINT)
                .addHeader("x-api-key", Config.LANGCHAIN_API_KEY)
                .addHeader("Langsmith-Project", project)
                .build();
        provider = SdkTracerProvider.builder()
                .setResource(Resource.getDefault().merge(Resource.create(
                        Attributes.of(AttributeKey.stringKey("service.name"), "guardrails"))))
                .addSpanProcessor(BatchSpanProcessor.builder(exporter).build())
                .build();
        otel = OpenTelemetrySdk.builder().setTracerProvider(provider).build();
        tracer = otel.getTracer("guardrails");
    }

    /** Flushes pending spans; call before the JVM exits. */
    public static void shutdown() {
        if (provider != null) {
            provider.shutdown().join(10, java.util.concurrent.TimeUnit.SECONDS);
        }
    }

    /** Runs body inside a span; kind is the LangSmith run type (chain, llm, tool). */
    public static <T> T trace(String name, String kind, Function<Span, T> body) {
        Span span = tracer.spanBuilder(name).setAttribute("langsmith.span.kind", kind).startSpan();
        try (Scope ignored = span.makeCurrent()) {
            return body.apply(span);
        } catch (RuntimeException e) {
            span.recordException(e);
            span.setStatus(StatusCode.ERROR);
            throw e;
        } finally {
            span.end();
        }
    }

    public static void trace(String name, String kind, Consumer<Span> body) {
        trace(name, kind, span -> {
            body.accept(span);
            return null;
        });
    }

    /** Adds a field shown under the run's metadata in LangSmith. */
    public static void metadata(Span span, String key, Object value) {
        span.setAttribute("langsmith.metadata." + key, String.valueOf(value));
    }
}
