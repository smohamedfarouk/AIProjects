package com.jupiter.langchaincommands.common;

import dev.langchain4j.model.chat.Capability;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.model.openai.OpenAiEmbeddingModel;

import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.time.Duration;

/**
 * Shared helpers used by every entry point of the project (each one does a single job - see README.md): OpenAI model
 * creation, console set-up and section headers.
 */
public final class Models {

    private Models() {
    }

    /** {@code ChatOpenAI(model=modelName)} with default settings. */
    public static ChatModel openAi(String modelName) {
        return openAi(modelName, false);
    }

    /**
     * Builds an OpenAI chat model.
     *
     * @param modelName   e.g. "gpt-5.4" or "gpt-4o-mini"
     * @param strictTools true makes OpenAI emit tool arguments that always match the tool's schema (valid JSON) -
     *                    used by the deep agent, which calls tools many times
     */
    public static ChatModel openAi(String modelName, boolean strictTools) {
        return OpenAiChatModel.builder()
                .apiKey(Config.OPENAI_API_KEY)
                .modelName(modelName)
                .strictTools(strictTools)
                .timeout(Duration.ofMinutes(3))
                .build();
    }

    /**
     * Prepares a demo run: switches the console to UTF-8 (replies contain characters like the rupee sign that the
     * Windows console codepage would garble) and stops with a clear message if OPENAI_API_KEY is missing.
     */
    public static void init() {
        initConsole();
        if (Config.OPENAI_API_KEY == null || Config.OPENAI_API_KEY.isBlank()) {
            System.err.println("OPENAI_API_KEY not found in environment or .env file.");
            System.exit(1);
        }
    }

    /**
     * Console set-up only - for entry points that make no OpenAI call (chunking, web loading): switches the console
     * to UTF-8 so non-ASCII text prints correctly on Windows. {@link #init()} does this plus the API-key check.
     */
    public static void initConsole() {
        System.setOut(new PrintStream(System.out, true, StandardCharsets.UTF_8));
    }

    /**
     * An OpenAI chat model in <b>strict JSON-schema mode</b> - the Java side of {@code llm.with_structured_output(...)}.
     * When an AI service method returns a record, LangChain4j sends the record's JSON schema and OpenAI must reply
     * with JSON that matches it exactly, so the reply can always be turned into the record.
     */
    public static ChatModel openAiJsonSchema(String modelName) {
        return OpenAiChatModel.builder()
                .apiKey(Config.OPENAI_API_KEY)
                .modelName(modelName)
                .supportedCapabilities(Capability.RESPONSE_FORMAT_JSON_SCHEMA)
                .strictJsonSchema(true)
                .timeout(Duration.ofMinutes(3))
                .build();
    }

    /**
     * {@code OpenAIEmbeddings(model=modelName)} - turns text into vectors, e.g. 1536 numbers for
     * "text-embedding-3-small".
     */
    public static EmbeddingModel openAiEmbeddings(String modelName) {
        return OpenAiEmbeddingModel.builder()
                .apiKey(Config.OPENAI_API_KEY)
                .modelName(modelName)
                .timeout(Duration.ofMinutes(2))
                .build();
    }

    /** Prints a section header so the output of each notebook cell is easy to find. */
    public static void section(String title) {
        System.out.println("\n========== " + title + " ==========\n");
    }
}
