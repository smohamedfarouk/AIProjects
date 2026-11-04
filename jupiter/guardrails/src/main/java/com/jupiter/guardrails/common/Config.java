package com.jupiter.guardrails.common;

import io.github.cdimascio.dotenv.Dotenv;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashSet;
import java.util.Set;

/**
 * Settings read from the environment, then from the nearest .env found by walking up
 * from the working directory and from the compiled classes' location (so it works from any run directory).
 */
public final class Config {

    public static final String OPENAI_API_KEY = get("OPENAI_API_KEY");
    public static final boolean LANGCHAIN_TRACING_V2 = Boolean.parseBoolean(get("LANGCHAIN_TRACING_V2"));
    public static final String LANGCHAIN_API_KEY = get("LANGCHAIN_API_KEY");
    public static final String LANGCHAIN_PROJECT = orDefault(get("LANGCHAIN_PROJECT"), "GuardRailsAgent ");

    private Config() {
    }

    public static String get(String name) {
        String value = System.getenv(name);
        if (value != null && !value.isBlank()) {
            return value;
        }
        for (Path dir : envDirs()) {
            value = Dotenv.configure().directory(dir.toString()).ignoreIfMissing().load().get(name, null);
            if (value != null && !value.isBlank()) {
                return value;
            }
        }
        return null;
    }

    private static Set<Path> envDirs() {
        Set<Path> dirs = new LinkedHashSet<>();
        addWithParents(dirs, Path.of("").toAbsolutePath());
        try {
            addWithParents(dirs, Path.of(Config.class.getProtectionDomain().getCodeSource().getLocation().toURI()));
        } catch (Exception ignored) {
            // code source unavailable; working-directory search still applies
        }
        dirs.removeIf(d -> !Files.isRegularFile(d.resolve(".env")));
        return dirs;
    }

    private static void addWithParents(Set<Path> dirs, Path start) {
        for (Path p = start; p != null; p = p.getParent()) {
            dirs.add(p);
        }
    }

    private static String orDefault(String value, String fallback) {
        return value == null || value.isBlank() ? fallback : value;
    }
}
