package com.bepec.mcp;

import io.modelcontextprotocol.json.McpJsonDefaults;
import io.modelcontextprotocol.server.McpServer;
import io.modelcontextprotocol.server.McpServerFeatures.SyncToolSpecification;
import io.modelcontextprotocol.server.transport.StdioServerTransportProvider;
import io.modelcontextprotocol.spec.McpSchema.CallToolResult;
import io.modelcontextprotocol.spec.McpSchema.ServerCapabilities;
import io.modelcontextprotocol.spec.McpSchema.Tool;

import java.io.FilterInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.function.Function;

/**
 * Small helpers that play the role of Python's FastMCP: {@code @mcp.tool()} and {@code mcp.run(transport="stdio")}.
 * In Python, FastMCP reads the function signature and docstring automatically; in Java we describe each tool's
 * name, description and JSON input schema explicitly.
 */
final class McpTools {

    private McpTools() {
    }

    /** One tool parameter: JSON type ("number", "integer", "string") and a short description. */
    record Param(String name, String type, String description, boolean required) {
        static Param required(String name, String type, String description) {
            return new Param(name, type, description, true);
        }

        static Param optional(String name, String type, String description) {
            return new Param(name, type, description, false);
        }
    }

    /**
     * Equivalent of {@code @mcp.tool()}: registers a tool whose handler receives the call's arguments and returns
     * text. Any exception becomes an error result, so the model sees the problem instead of the server crashing.
     */
    static SyncToolSpecification tool(String name, String description, List<Param> params,
                                      Function<Map<String, Object>, String> handler) {
        Map<String, Object> properties = new LinkedHashMap<>();
        List<String> required = params.stream().filter(Param::required).map(Param::name).toList();
        for (Param p : params) {
            properties.put(p.name(), Map.of("type", p.type(), "description", p.description()));
        }
        Tool tool = Tool.builder()
                .name(name)
                .description(description)
                .inputSchema(Map.of("type", "object", "properties", properties, "required", required))
                .build();

        return SyncToolSpecification.builder()
                .tool(tool)
                .callHandler((exchange, request) -> {
                    Map<String, Object> args = request.arguments() == null ? Map.of() : request.arguments();
                    try {
                        return CallToolResult.builder().addTextContent(handler.apply(args)).build();
                    } catch (RuntimeException e) {
                        return CallToolResult.builder().addTextContent("Error: " + e.getMessage()).isError(true).build();
                    }
                })
                .build();
    }

    /**
     * Equivalent of {@code mcp.run(transport="stdio")}: serves the tools over this process's stdin/stdout.
     * <p>
     * How it works:
     * <ol>
     *   <li>The client (in the agent process) started this process and holds both ends of its pipes.
     *       Each line it writes to our <b>stdin</b> is one JSON-RPC request; each line we write to <b>stdout</b> is
     *       one response. Nothing else may ever be printed to stdout - it would corrupt the protocol
     *       (that is why the project uses slf4j-nop: no log lines on stdout).</li>
     *   <li>The MCP SDK's {@link StdioServerTransportProvider} reads stdin on a background thread and answers:
     *       {@code initialize} (handshake: server name/version and capabilities - here "tools"),
     *       {@code tools/list} (the tool specifications) and {@code tools/call} (runs the matching handler).</li>
     *   <li>The main thread just waits. When the client is finished it closes our stdin; reading then hits
     *       end-of-stream, we release the wait and exit. This is the normal MCP stdio shutdown - without it the
     *       client's close() would hang on Windows waiting for our stdout pipe to close.</li>
     * </ol>
     *
     * @param serverName name reported in the {@code initialize} handshake, e.g. "BeginnerTools"
     * @param tools      the tools to expose (built with {@link #tool})
     */
    static void serveStdio(String serverName, List<SyncToolSpecification> tools) throws InterruptedException {
        CountDownLatch clientDisconnected = new CountDownLatch(1);
        InputStream stdin = new FilterInputStream(System.in) {
            @Override
            public int read() throws IOException {
                return eof(super.read());
            }

            @Override
            public int read(byte[] b, int off, int len) throws IOException {
                return eof(super.read(b, off, len));
            }

            private int eof(int n) {
                if (n == -1) {
                    clientDisconnected.countDown();
                }
                return n;
            }
        };

        var transport = new StdioServerTransportProvider(McpJsonDefaults.getMapper(), stdin, System.out);
        McpServer.sync(transport)
                .serverInfo(serverName, "1.0.0")
                .capabilities(ServerCapabilities.builder().tools(true).build())
                .tools(tools)
                .build();

        clientDisconnected.await();
        System.exit(0); // the SDK's worker threads are not daemons, so end the process explicitly
    }

    /**
     * Reads a numeric argument from a tool call. JSON numbers arrive as Integer or Double (8 vs 8.5), so both are
     * accepted; a number sent as text ("8") also works. Missing values throw, which becomes an error result.
     */
    static double number(Map<String, Object> args, String key) {
        Object value = args.get(key);
        if (value instanceof Number n) {
            return n.doubleValue();
        }
        if (value instanceof String s && !s.isBlank()) {
            return Double.parseDouble(s.strip());
        }
        throw new IllegalArgumentException("Missing numeric argument: " + key);
    }

    /** Reads a required text argument (e.g. "city") from a tool call; missing or blank values throw. */
    static String text(Map<String, Object> args, String key) {
        Object value = args.get(key);
        if (value == null || value.toString().isBlank()) {
            throw new IllegalArgumentException("Missing text argument: " + key);
        }
        return value.toString().strip();
    }
}
