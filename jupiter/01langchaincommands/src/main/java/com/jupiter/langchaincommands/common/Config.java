package com.jupiter.langchaincommands.common;

import io.github.cdimascio.dotenv.Dotenv;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashSet;
import java.util.Set;

/**
 * API keys for all entry points ({@code search.DeepAgentWithTavilyMain} uses both keys; every other entry point
 * only OPENAI_API_KEY). Each value is read from the environment first, then from the nearest .env file found by
 * walking up from the working directory and from the compiled classes' location - so it works whether you start it
 * from Maven, from IntelliJ, or from another folder, and finds e.g. {@code jupiter/.env}.
 */
public final class Config {

    public static final String OPENAI_API_KEY = get("OPENAI_API_KEY");
    public static final String TAVILY_API_KEY = get("TAVILY_API_KEY");

    private Config() {
    }

    /**
     * Looks up one setting.
     * <ol>
     *   <li>The environment variable, if set and not blank.</li>
     *   <li>Otherwise each .env file from the nearest folder upwards; the first non-blank value wins.</li>
     * </ol>
     *
     * @param name e.g. "OPENAI_API_KEY"
     * @return the value, or null if it is not set anywhere (the demos then stop with a clear message)
     */
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

}
