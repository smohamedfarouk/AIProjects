package com.bepec.deepagents.agent;

import dev.langchain4j.agent.tool.P;
import dev.langchain4j.agent.tool.Tool;

import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;

/**
 * In-memory file system shared between a deep agent and its subagents
 * (mirrors the ls / read_file / write_file / edit_file tools of the Python deepagents package).
 */
public class VirtualFileSystemTools {

    private final Map<String, String> files = new ConcurrentHashMap<>();

    public Map<String, String> files() {
        return new TreeMap<>(files);
    }

    @Tool(name = "ls", value = "List all files in the agent's virtual file system.")
    public String ls() {
        return files.isEmpty() ? "(no files)" : String.join("\n", new TreeMap<>(files).keySet());
    }

    @Tool(name = "read_file", value = "Read the contents of a file from the virtual file system.")
    public String readFile(@P("path of the file, e.g. 'seeds.md'") String path) {
        String content = files.get(normalize(path));
        return content != null ? content : "Error: file '" + path + "' not found";
    }

    @Tool(name = "write_file", value = "Write (create or overwrite) a file in the virtual file system.")
    public String writeFile(@P("path of the file, e.g. 'seeds.md'") String path,
                            @P("full content of the file") String content) {
        files.put(normalize(path), content);
        return "Wrote " + content.length() + " characters to " + path;
    }

    @Tool(name = "edit_file", value = "Edit a file by replacing an exact string with a new string.")
    public String editFile(@P("path of the file") String path,
                           @P("exact text to replace") String oldString,
                           @P("replacement text") String newString) {
        String key = normalize(path);
        String content = files.get(key);
        if (content == null) {
            return "Error: file '" + path + "' not found";
        }
        if (!content.contains(oldString)) {
            return "Error: text to replace not found in " + path;
        }
        files.put(key, content.replace(oldString, newString));
        return "Edited " + path;
    }

    private static String normalize(String path) {
        String p = path.trim().replace('\\', '/');
        return p.startsWith("/") ? p.substring(1) : p;
    }
}
