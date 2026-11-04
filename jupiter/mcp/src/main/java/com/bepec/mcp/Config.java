package com.bepec.mcp;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import io.github.cdimascio.dotenv.Dotenv;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;

/** load_dotenv() + init_chat_model("openai:gpt-4o-mini"). */
public final class Config {

    public static final String MODEL_NAME = "gpt-4o-mini";

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

    /**
     * Builds the OpenAI chat model used by the agent - the Java version of {@code init_chat_model("openai:gpt-4o-mini")}.
     * The API key comes from the OPENAI_API_KEY environment variable or a .env file (see {@link #env}); without it
     * this throws a clear error before anything else starts. No temperature is set, so the OpenAI default is used,
     * as in agent.py.
     */
    public static ChatModel llm() {
        String key = env("OPENAI_API_KEY");
        if (key == null || key.isBlank()) {
            throw new IllegalStateException(
                    "OPENAI_API_KEY not found. Create a .env file with: OPENAI_API_KEY=your_key_here");
        }
        return OpenAiChatModel.builder()
                .apiKey(key)
                .modelName(MODEL_NAME)
                .timeout(Duration.ofMinutes(2))
                .build();
    }
}
