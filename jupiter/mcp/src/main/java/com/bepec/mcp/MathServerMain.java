package com.bepec.mcp;

import com.bepec.mcp.McpTools.Param;
import io.modelcontextprotocol.server.McpServerFeatures.SyncToolSpecification;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;

/**
 * Java port of math_server.py - an MCP server named "BeginnerTools" with three tools.
 * <p>
 * MCP - connects LLM models with tools (functions) or external data.
 * <ul>
 *   <li>MCP Host - the AI application that manages the model ({@link AgentMain})</li>
 *   <li>MCP Server - code that exposes functions (tools) or external data (this class)</li>
 *   <li>MCP Client - component inside the host that connects it to a server ({@link MultiServerMcpClient})</li>
 *   <li>MCP Transport - the communication channel (here: stdio)</li>
 * </ul>
 * Started by the agent as a child process; it can also be run alone:
 * {@code java -cp <classpath> com.bepec.mcp.MathServerMain}.
 */
public final class MathServerMain {

    private static final DateTimeFormatter TIME = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    private MathServerMain() {
    }

    /** Add two numbers. */
    static double add(double a, double b) {
        return a + b;
    }

    /** Multiply two numbers. */
    static double multiply(double a, double b) {
        return a * b;
    }

    /** Return the current date and time. */
    static String currentTime() {
        return LocalDateTime.now().format(TIME);
    }

    /**
     * Describes the three tools for MCP - the Java version of decorating each function with {@code @mcp.tool()}.
     * <p>
     * For each tool we give: the <b>name</b> the model will call, a <b>description</b> (the model reads it to decide
     * when to use the tool, like a Python docstring), the <b>parameters</b> as a JSON schema (add and multiply need
     * numbers "a" and "b"; current_time needs none), and the <b>handler</b> that reads the arguments, calls the Java
     * method above and returns the answer as text (e.g. 8.0 becomes "8.0").
     *
     * @return the tool definitions served by {@link #main}
     */
    static List<SyncToolSpecification> tools() {
        List<Param> ab = List.of(
                Param.required("a", "number", "First number"),
                Param.required("b", "number", "Second number"));
        return List.of(
                McpTools.tool("add", "Add two numbers.", ab,
                        args -> String.valueOf(add(McpTools.number(args, "a"), McpTools.number(args, "b")))),
                McpTools.tool("multiply", "Multiply two numbers.", ab,
                        args -> String.valueOf(multiply(McpTools.number(args, "a"), McpTools.number(args, "b")))),
                McpTools.tool("current_time", "Return the current date and time.", List.of(),
                        args -> currentTime()));
    }

    /**
     * Entry point of the MCP server process - the Java version of {@code mcp.run(transport="stdio")}.
     * <p>
     * You normally do not run this yourself: {@link MultiServerMcpClient} starts it as a child process of the
     * agent. From then on:
     * <ol>
     *   <li>The agent sends JSON-RPC requests on this process's stdin, one per line.</li>
     *   <li>{@code initialize} - the handshake; we answer with our name "BeginnerTools" and that we offer tools.</li>
     *   <li>{@code tools/list} - we answer with the definitions from {@link #tools()}.</li>
     *   <li>{@code tools/call} - e.g. multiply with {"a": 8, "b": 12}; we run the handler and answer "96.0" on
     *       stdout.</li>
     *   <li>When the agent finishes it closes our stdin, and this process exits.</li>
     * </ol>
     * Because stdout carries the protocol, this class must never print anything with System.out.
     *
     * @param args not used
     */
    public static void main(String[] args) throws InterruptedException {
        McpTools.serveStdio("BeginnerTools", tools());
    }
}
