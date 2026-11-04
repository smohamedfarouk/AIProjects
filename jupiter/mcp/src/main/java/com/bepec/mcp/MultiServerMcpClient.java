package com.bepec.mcp;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.mcp.client.DefaultMcpClient;
import dev.langchain4j.mcp.client.McpClient;
import dev.langchain4j.mcp.client.transport.stdio.StdioMcpTransport;
import dev.langchain4j.service.tool.ToolExecutionResult;

import java.net.URISyntaxException;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.io.File;

/**
 * Java equivalent of langchain_mcp_adapters' {@code MultiServerMCPClient}: starts each MCP server as a child
 * process (stdio transport), collects all their tools, and routes every tool call to the server that owns it.
 */
public final class MultiServerMcpClient implements AutoCloseable {

    private final List<McpClient> clients = new ArrayList<>();
    private final Map<String, McpClient> toolOwners = new LinkedHashMap<>();
    private final List<ToolSpecification> tools = new ArrayList<>();

    /**
     * Starts every server and loads its tools - the Java version of creating {@code MultiServerMCPClient({...})} and
     * calling {@code await client.get_tools()}.
     * <p>
     * For each server:
     * <ol>
     *   <li>{@link StdioMcpTransport} starts the server's main class as a child process ({@link #javaCommand}).</li>
     *   <li>{@link DefaultMcpClient} performs the MCP {@code initialize} handshake over the process's stdin/stdout.</li>
     *   <li>{@code listTools()} sends {@code tools/list}; each tool is remembered together with the client that owns
     *       it, so {@link #execute} later knows which server to call.</li>
     * </ol>
     * If any server fails to start, the ones already started are closed again before the error is rethrown.
     *
     * @param servers server name -> Java main class of that MCP server, e.g. {@code "beginner_tools" -> MathServerMain}
     */
    public MultiServerMcpClient(Map<String, Class<?>> servers) {
        try {
            for (Map.Entry<String, Class<?>> server : servers.entrySet()) {
                McpClient client = new DefaultMcpClient.Builder()
                        .key(server.getKey())
                        .transport(new StdioMcpTransport.Builder()
                                .command(javaCommand(server.getValue()))
                                .logEvents(false)
                                .build())
                        .build();
                clients.add(client);
                // tools/list: ask the server which tools it has (name, description, input schema)
                for (ToolSpecification tool : client.listTools()) {
                    toolOwners.put(tool.name(), client);
                    tools.add(tool);
                }
            }
        } catch (RuntimeException e) {
            close();
            throw e;
        }
    }

    /**
     * All tools from all servers, as LangChain4j tool specifications (name, description, JSON schema of the
     * arguments). The agent sends these to the model with every request - the equivalent of the {@code tools} list
     * returned by {@code await client.get_tools()}.
     */
    public List<ToolSpecification> getTools() {
        return List.copyOf(tools);
    }

    /**
     * Runs one tool call requested by the model ({@code tools/call} in MCP).
     * <ol>
     *   <li>Find the server that owns the tool (remembered while listing tools at start-up).</li>
     *   <li>Send the call - tool name plus the model's JSON arguments - over that server's stdin.</li>
     *   <li>Return the text the server writes back on its stdout.</li>
     * </ol>
     * Problems (unknown tool, arguments that fail the tool's JSON schema, an exception inside the tool) are
     * returned as {@code "Error: ..."} text instead of being thrown. Like LangGraph's ToolNode, this lets the model
     * read the error and retry with corrected arguments, instead of the whole agent run crashing.
     *
     * @param request the tool call from the model's reply: id, tool name and JSON arguments
     * @return the tool's result text, or an error message the model can read
     */
    public String execute(ToolExecutionRequest request) {
        McpClient owner = toolOwners.get(request.name());
        if (owner == null) {
            return "Error: unknown tool '" + request.name() + "'";
        }
        try {
            ToolExecutionResult result = owner.executeTool(request);
            return result.resultText();
        } catch (RuntimeException e) {
            return "Error: " + e.getMessage();
        }
    }

    /**
     * Stops every server: closing a client closes the server's stdin, the server sees end-of-input and exits
     * (see {@link McpTools#serveStdio}), then the client's pipes are released. Errors are ignored because this runs
     * during shutdown.
     */
    @Override
    public void close() {
        for (McpClient client : clients) {
            try {
                client.close(); // also stops the server process
            } catch (Exception ignored) {
                // best effort while shutting down
            }
        }
        clients.clear();
    }

    /**
     * Equivalent of {@code "command": sys.executable, "args": [SERVER_PATH]}: the same java binary and the same
     * classpath as this program, running the server's main class.
     */
    static List<String> javaCommand(Class<?> mainClass) {
        String java = Path.of(System.getProperty("java.home"), "bin", "java").toString();
        return List.of(java, "-cp", classpath(), mainClass.getName());
    }

    /**
     * {@code mvn exec:java} runs inside Maven with its own class loader, so java.class.path would be Maven's;
     * read the real classpath from the class loader when it exposes one.
     */
    static String classpath() {
        ClassLoader loader = MultiServerMcpClient.class.getClassLoader();
        if (loader instanceof URLClassLoader urlLoader && urlLoader.getURLs().length > 0) {
            List<String> paths = new ArrayList<>();
            for (URL url : urlLoader.getURLs()) {
                try {
                    paths.add(Path.of(url.toURI()).toString());
                } catch (URISyntaxException | IllegalArgumentException e) {
                    // skip entries that are not local files
                }
            }
            return String.join(File.pathSeparator, paths);
        }
        return System.getProperty("java.class.path");
    }
}
