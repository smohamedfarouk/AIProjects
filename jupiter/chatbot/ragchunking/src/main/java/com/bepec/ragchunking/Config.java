package com.bepec.ragchunking;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.cohere.CohereScoringModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.model.openai.OpenAiEmbeddingModel;
import dev.langchain4j.model.scoring.ScoringModel;
import io.github.cdimascio.dotenv.Dotenv;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;

/**
 * Settings and models - the Java version of the notebook's {@code ChatOpenAI(...)}, {@code OpenAIEmbeddings(...)}
 * and {@code CohereRerank(...)}.
 * <p>
 * The notebook pastes the OpenAI key straight into the code. Here keys come from the environment or a {@code .env}
 * file (OPENAI_API_KEY, COHERE_API_KEY), so they never end up in source control.
 */
public final class Config {

    public static final String CHAT_MODEL = "gpt-4o-mini";
    public static final String EMBEDDING_MODEL = "text-embedding-3-small";
    public static final String RERANK_MODEL = "rerank-english-v3.0";

    private static final Dotenv DOTENV = loadDotenv();

    private Config() {
    }

    /** Looks for .env in the working directory and its parents, then upward from the compiled classes. */
    private static Dotenv loadDotenv() {
        Path dir = findEnvDir(Path.of("").toAbsolutePath());
        if (dir == null) {
            dir = findEnvDir(codeLocation());
        }
        var cfg = Dotenv.configure().ignoreIfMissing();
        if (dir != null) {
            cfg = cfg.directory(dir.toString());
        }
        return cfg.load();
    }

    private static Path findEnvDir(Path start) {
        Path dir = start;
        while (dir != null && !Files.exists(dir.resolve(".env"))) {
            dir = dir.getParent();
        }
        return dir;
    }

    private static Path codeLocation() {
        try {
            return Path.of(Config.class.getProtectionDomain().getCodeSource().getLocation().toURI());
        } catch (Exception e) {
            return null;
        }
    }

    /** Environment variables win over values from .env. */
    public static String env(String name) {
        String value = System.getenv(name);
        return value != null && !value.isBlank() ? value : DOTENV.get(name);
    }

    /** True when the setting is present and not blank - used to skip demos whose API key is missing. */
    public static boolean has(String name) {
        String value = env(name);
        return value != null && !value.isBlank();
    }

    private static String require(String name) {
        if (!has(name)) {
            throw new IllegalStateException(name + " not found. Add it to a .env file: " + name + "=your_key_here");
        }
        return env(name);
    }

    /** {@code ChatOpenAI(model="gpt-4o-mini", temperature=0)}. */
    public static ChatModel llm() {
        return OpenAiChatModel.builder()
                .apiKey(require("OPENAI_API_KEY"))
                .modelName(CHAT_MODEL)
                .temperature(0.0)
                .timeout(Duration.ofMinutes(2))
                .build();
    }

    /** {@code OpenAIEmbeddings(model="text-embedding-3-small")}. */
    public static EmbeddingModel embeddings() {
        return OpenAiEmbeddingModel.builder()
                .apiKey(require("OPENAI_API_KEY"))
                .modelName(EMBEDDING_MODEL)
                .timeout(Duration.ofMinutes(2))
                .build();
    }

    /** {@code CohereRerank(model="rerank-english-v3.0")} - a cross-encoder that scores (query, document) pairs. */
    public static ScoringModel reranker() {
        return CohereScoringModel.builder()
                .apiKey(require("COHERE_API_KEY"))
                .modelName(RERANK_MODEL)
                .timeout(Duration.ofMinutes(2))
                .build();
    }
}
