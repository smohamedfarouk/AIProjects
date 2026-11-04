package com.bepec.ragchatbot;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.model.openai.OpenAiEmbeddingModel;
import io.github.cdimascio.dotenv.Dotenv;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;

/**
 * Settings and OpenAI models - the Java version of {@code load_dotenv()}, {@code ChatOpenAI(...)} and
 * {@code OpenAIEmbeddings(...)} from the notebook.
 */
public final class Config {

    /** Default PDF, same name as in the notebook; override with the first program argument or PDF_PATH. */
    public static final String DEFAULT_PDF = "AI Agenda_Updated'26.pdf";
    public static final String CHAT_MODEL = "gpt-4o-mini";
    public static final String EMBEDDING_MODEL = "text-embedding-3-small";

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

    private static String apiKey() {
        String key = env("OPENAI_API_KEY");
        if (key == null || key.isBlank()) {
            throw new IllegalStateException(
                    "OPENAI_API_KEY not found. Create a .env file with: OPENAI_API_KEY=your_key_here");
        }
        return key;
    }

    /**
     * The answering model: {@code ChatOpenAI(model="gpt-4o-mini", temperature=0)}. Temperature 0 makes answers as
     * repeatable as possible - right for "answer only from the document" questions.
     */
    public static ChatModel llm() {
        return OpenAiChatModel.builder()
                .apiKey(apiKey())
                .modelName(CHAT_MODEL)
                .temperature(0.0)
                .timeout(Duration.ofMinutes(2))
                .build();
    }

    /**
     * The embedding model: {@code OpenAIEmbeddings(model="text-embedding-3-small")}. It turns a piece of text into
     * a list of 1536 numbers (a vector); texts with similar meaning get vectors that point in similar directions.
     */
    public static EmbeddingModel embeddings() {
        return OpenAiEmbeddingModel.builder()
                .apiKey(apiKey())
                .modelName(EMBEDDING_MODEL)
                .timeout(Duration.ofMinutes(2))
                .build();
    }
}
