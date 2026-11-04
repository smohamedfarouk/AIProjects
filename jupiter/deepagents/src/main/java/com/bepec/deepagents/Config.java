package com.bepec.deepagents;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import io.github.cdimascio.dotenv.Dotenv;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;

/** load_dotenv() + ChatOpenAI(model='gpt-5.4-mini', temperature=0). */
public final class Config {

    public static final String MODEL_NAME = "gpt-5.4-mini";

    private static final Dotenv DOTENV = loadDotenv();

    private Config() {
    }

    /**
     * Looks for a .env file in the working directory and its parents (e.g. jupiter/.env); if none is
     * found there, searches upward from the compiled classes location so the lookup does not depend
     * on the IDE's working directory.
     */
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

    public static ChatModel llm() {
        String key = env("OPENAI_API_KEY");
        if (key == null || key.isBlank()) {
            throw new IllegalStateException("OPENAI_API_KEY is not set (environment or .env)");
        }
        return OpenAiChatModel.builder()
                .apiKey(key)
                .modelName(MODEL_NAME)
                .temperature(0.0)
                .timeout(Duration.ofMinutes(3))
                .build();
    }
}
