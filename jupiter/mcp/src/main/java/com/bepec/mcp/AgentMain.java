package com.bepec.mcp;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * BEPEC MCP Agent - main class. Java port of agent.py (MCP host + client + LangGraph agent loop).
 * <p>
 * Run: {@code mvn compile exec:java} (needs OPENAI_API_KEY in the environment or a .env file).
 *
 * <h2>Key idea: the LLM cannot run code - MCP lets it use tools that live in another program</h2>
 * gpt-4o-mini can only produce text. It cannot add numbers reliably or read the clock. MCP (Model Context
 * Protocol) is a standard way to plug tools (functions) and external data into an LLM app. The tools live in a
 * separate program (an MCP server); the app discovers them at start-up and runs them whenever the model asks.
 *
 * <h2>MCP vocabulary</h2>
 * <pre>
 * | Term       | In this project                                                              |
 * |------------|------------------------------------------------------------------------------|
 * | MCP Host   | AgentMain (this class) - the AI application that owns the model and the loop      |
 * | MCP Client | MultiServerMcpClient - connects the host to one or more servers               |
 * | MCP Server | MathServerMain - exposes the tools add, multiply, current_time                    |
 * | Transport  | stdio - the client starts the server as a child process and they exchange     |
 * |            | JSON-RPC messages over its stdin / stdout                                     |
 * </pre>
 *
 * <h2>What happens when you run it</h2>
 * <ol>
 *   <li><b>Start the server.</b> {@link MultiServerMcpClient} launches
 *       {@code java -cp <same classpath> com.bepec.mcp.MathServerMain} as a child process (stdio transport) -
 *       the Java version of {@code "command": sys.executable, "args": [SERVER_PATH]}.</li>
 *   <li><b>Discover the tools.</b> The client sends {@code tools/list}. The server answers with each tool's name,
 *       description and JSON input schema, e.g. {@code multiply(a: number, b: number)}.<br>
 *       Output: {@code Tools loaded from MCP: [add, multiply, current_time]}</li>
 *   <li><b>Give the tools to the model.</b> Every {@link ChatRequest} carries those tool specifications
 *       ({@code llm.bind_tools(tools)} in Python), so the model knows which tools exist and can answer with a
 *       tool call instead of plain text.</li>
 *   <li><b>Run the agent loop</b> ({@link #run}) - the Java version of the LangGraph graph:
 * <pre>
 *     START --> [agent] --tool calls?--yes--> [tools] --+
 *                  ^                                     |
 *                  +-------------------------------------+
 *                  |
 *                  +--no tool calls--> END (final answer)
 * </pre>
 *       <ul>
 *         <li>"agent" step: call the model with all messages so far.</li>
 *         <li>tools_condition: if the reply contains tool calls go to "tools", otherwise stop.</li>
 *         <li>"tools" step (ToolNode): run each requested tool. Each call travels through the MCP client
 *             ({@code tools/call}) to MathServerMain; the answer comes back as a {@link ToolExecutionResultMessage}.</li>
 *         <li>tools -> agent: the model sees the results and decides what to do next.</li>
 *       </ul>
 *       The {@code messages} list plays the role of LangGraph's MessagesState (user, AI and tool messages).
 *       {@link #MAX_STEPS} stops a model that keeps calling tools forever.</li>
 *   <li><b>Print every message</b> - like {@code pretty_print()}, including each tool call and its result.</li>
 *   <li><b>Clean up.</b> Closing the client stops the server process.</li>
 * </ol>
 *
 * <h2>Example: "What is (3 + 5) multiplied by 12? Also, what time is it right now?"</h2>
 * <pre>
 * | Step | Who         | What happens                                                    |
 * |------|-------------|-----------------------------------------------------------------|
 * | 1    | Human       | Asks the question                                               |
 * | 2    | AI          | Tool calls: add(a=3, b=5) and current_time()                    |
 * | 3    | Tools (MCP) | MathServerMain returns 8.0 and e.g. "2026-10-31 22:40:12"           |
 * | 4    | AI          | Tool call: multiply(a=8, b=12) - it needed the 8 first          |
 * | 5    | Tools (MCP) | MathServerMain returns 96.0                                         |
 * | 6    | AI          | No more tool calls -> final answer: "96, and the time is ..."   |
 * </pre>
 * The model chooses the exact calls, so the steps can vary slightly between runs.
 *
 * <h2>Classes in this project</h2>
 * <pre>
 * | Class                | What it does                                          | Python equivalent        |
 * |----------------------|-------------------------------------------------------|--------------------------|
 * | AgentMain            | Main class: MCP host + agent loop (this class)        | agent.py                 |
 * | MultiServerMcpClient | Starts servers as child processes, lists their tools, | MultiServerMCPClient     |
 * |                      | routes each tool call to the right server             |                          |
 * | MathServerMain       | MCP server "BeginnerTools": add, multiply,            | math_server.py           |
 * |                      | current_time                                          |                          |
 * | WeatherServerMain    | MCP server "WeatherTools": get_current_weather,       | weather_server.py        |
 * |                      | get_forecast (Open-Meteo, no key needed)              |                          |
 * | McpTools             | Describes a tool (name, description, JSON schema) and | FastMCP, @mcp.tool(),    |
 * |                      | serves tools over stdio                               | mcp.run("stdio")         |
 * | Config               | Loads .env and builds the OpenAI model (gpt-4o-mini)  | load_dotenv() +          |
 * |                      |                                                       | init_chat_model()        |
 * </pre>
 *
 * <h2>Using the weather tools too</h2>
 * Like agent.py, only MathServerMain is connected. To add the weather tools, add one entry to {@link #SERVERS}:
 * {@code "weather_tools", WeatherServerMain.class} - the agent then also sees get_current_weather and get_forecast.
 *
 * <h2>Limits</h2>
 * There is no memory between runs: each run asks one fixed question and exits. The tool servers run locally as
 * child processes and stop when the agent finishes.
 */
public final class AgentMain {

    static final String QUESTION = "What is (3 + 5) multiplied by 12? Also, what time is it right now?";

    /** Safety limit on agent -> tools rounds. */
    static final int MAX_STEPS = 10;

    /** MCP servers to start: name -> main class (the Python client dict). */
    static final Map<String, Class<?>> SERVERS = new LinkedHashMap<>(Map.of(
            "beginner_tools", MathServerMain.class));

    private AgentMain() {
    }

    /**
     * Entry point - runs the whole MCP demo once (the Java version of {@code asyncio.run(main())} in agent.py).
     * <ol>
     *   <li><b>Create the model</b> ({@link Config#llm()}). Done first so a missing OPENAI_API_KEY fails
     *       immediately, before any server process is started.</li>
     *   <li><b>Start the MCP servers and load their tools</b> - {@code new MultiServerMcpClient(SERVERS)} launches
     *       each server in {@link #SERVERS} as a child process, does the MCP handshake, and asks each one for its
     *       tools. Prints {@code Tools loaded from MCP: [add, multiply, current_time]}.</li>
     *   <li><b>Run the agent loop</b> on {@link #QUESTION} with {@link #run} - the model calls tools as often as it
     *       needs until it can give a final answer.</li>
     *   <li><b>Print the conversation</b> - every message (question, tool calls, tool results, final answer) in
     *       the same style as LangChain's {@code pretty_print()}.</li>
     *   <li><b>Shut down</b> - try-with-resources closes the client, which closes each server's stdin; the servers
     *       see end-of-input and exit (see {@link McpTools#serveStdio}).</li>
     * </ol>
     *
     * @param args not used
     */
    public static void main(String[] args) {
        ChatModel llm = Config.llm(); // fail fast if the API key is missing, before starting any server

        try (MultiServerMcpClient client = new MultiServerMcpClient(SERVERS)) {
            List<ToolSpecification> tools = client.getTools();
            System.out.println("Tools loaded from MCP: " + tools.stream().map(ToolSpecification::name).toList());

            List<ChatMessage> messages = run(llm, client, tools, QUESTION);

            messages.forEach(m -> System.out.println(prettyPrint(m)));
        }
    }

    /**
     * The agent loop - the Java version of the compiled LangGraph graph ({@code app.ainvoke(...)}).
     * <p>
     * Each round:
     * <ol>
     *   <li><b>"agent" node</b> - send the model ALL messages so far plus the tool specifications. The model
     *       either answers in text, or replies with one or more tool calls (tool name + JSON arguments).</li>
     *   <li><b>tools_condition</b> - no tool calls means the model is done: return the messages.</li>
     *   <li><b>"tools" node</b> - run every requested tool through MCP and add each result as a
     *       {@link ToolExecutionResultMessage} (linked to its call by id), then go back to step 1 so the model
     *       can read the results.</li>
     * </ol>
     * Example: round 1 the model asks for add(3, 5) and current_time(); round 2 it sees 8.0 and asks for
     * multiply(8, 12); round 3 it sees 96.0 and writes the final answer.
     * <p>
     * The loop stops after {@link #MAX_STEPS} rounds, so a model that keeps calling tools cannot run forever.
     *
     * @param llm      the chat model (OpenAI in production, a scripted fake in tests)
     * @param client   routes each tool call to the MCP server that owns the tool
     * @param tools    the tool specifications loaded from the MCP servers
     * @param question the user's question
     * @return every message of the run: question, AI replies, tool results and the final answer
     */
    static List<ChatMessage> run(ChatModel llm, MultiServerMcpClient client, List<ToolSpecification> tools,
                                 String question) {
        List<ChatMessage> messages = new ArrayList<>();
        messages.add(UserMessage.from(question));

        for (int step = 1; step <= MAX_STEPS; step++) {
            // "agent" node
            AiMessage ai = llm.chat(ChatRequest.builder()
                    .messages(messages)
                    .toolSpecifications(tools)
                    .build()).aiMessage();
            messages.add(ai);

            // tools_condition
            if (!ai.hasToolExecutionRequests()) {
                return messages;
            }

            // "tools" node
            for (ToolExecutionRequest call : ai.toolExecutionRequests()) {
                messages.add(ToolExecutionResultMessage.from(call, client.execute(call)));
            }
        }
        messages.add(AiMessage.from("Stopped after " + MAX_STEPS + " steps without a final answer."));
        return messages;
    }

    /**
     * Java version of LangChain's {@code message.pretty_print()}: a centred title line, then the content.
     * <ul>
     *   <li>Human Message - the question</li>
     *   <li>Ai Message - the model's text and/or its tool calls (name, call id, JSON arguments)</li>
     *   <li>Tool Message - which tool ran and the result text the MCP server returned</li>
     * </ul>
     *
     * @param message any message from {@link #run}
     * @return the formatted text, ready to print
     */
    static String prettyPrint(ChatMessage message) {
        String title;
        StringBuilder body = new StringBuilder();
        switch (message) {
            case UserMessage u -> {
                title = "Human Message";
                body.append(u.singleText());
            }
            case AiMessage a -> {
                title = "Ai Message";
                if (a.text() != null) {
                    body.append(a.text());
                }
                if (a.hasToolExecutionRequests()) {
                    body.append(body.isEmpty() ? "" : "\n").append("Tool Calls:");
                    for (ToolExecutionRequest call : a.toolExecutionRequests()) {
                        body.append("\n  ").append(call.name()).append(" (").append(call.id()).append(")")
                                .append("\n    Args: ").append(call.arguments());
                    }
                }
            }
            case ToolExecutionResultMessage t -> {
                title = "Tool Message";
                body.append("Name: ").append(t.toolName()).append("\n\n").append(t.text());
            }
            default -> {
                title = message.type().toString();
                body.append(message);
            }
        }
        String pad = "=".repeat(Math.max(0, (80 - title.length() - 2) / 2));
        return pad + " " + title + " " + pad + "\n\n" + body;
    }
}
