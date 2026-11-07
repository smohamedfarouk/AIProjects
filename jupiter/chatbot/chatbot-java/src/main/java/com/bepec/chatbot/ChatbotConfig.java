package com.bepec.chatbot;

import dev.langchain4j.model.TokenCountEstimator;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.model.openai.OpenAiTokenCountEstimator;
import io.github.cdimascio.dotenv.Dotenv;
import org.springframework.boot.system.ApplicationHome;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

/**
 * Application configuration: loads the OpenAI API key from .env (or the
 * environment) and builds the chat model and token counter beans.
 */
@Configuration
public class ChatbotConfig {

    static final String MODEL_NAME = "gpt-4o-mini";
    static final double TEMPERATURE = 0.3;
    static final int MAX_TOKENS = 1024;

    @Bean
    public ChatModel chatModel() {
        return OpenAiChatModel.builder()
                .apiKey(loadApiKey())
                .modelName(MODEL_NAME)
                .temperature(TEMPERATURE)
                .maxTokens(MAX_TOKENS)
                .build();
    }

    @Bean
    public TokenCountEstimator tokenCountEstimator() {
        return new OpenAiTokenCountEstimator(MODEL_NAME);
    }

    private static String loadApiKey() {
        Dotenv dotenv = Dotenv.configure()
                .directory(findEnvDirectory().toString())
                .ignoreIfMissing()
                .load();
        String apiKey = dotenv.get("OPENAI_API_KEY"); // falls back to system env vars
        if (apiKey == null || apiKey.isBlank()) {
            throw new IllegalStateException(
                    "OPENAI_API_KEY not found. Create a .env file with: OPENAI_API_KEY=your_key_here");
        }
        return apiKey;
    }

    /**
     * Nearest directory containing a .env file. Walks up from the working dir first,
     * then from the app's own location (the jar or target/classes), so it finds
     * jupiter/.env no matter which folder the app is started from.
     */
    private static Path findEnvDirectory() {
        Path workingDir = Paths.get("").toAbsolutePath();
        Path appDir = new ApplicationHome(ChatbotConfig.class).getDir().toPath().toAbsolutePath();
        for (Path start : List.of(workingDir, appDir)) {
            for (Path dir = start; dir != null; dir = dir.getParent()) {
                if (Files.isRegularFile(dir.resolve(".env"))) {
                    return dir;
                }
            }
        }
        return workingDir;
    }
}
