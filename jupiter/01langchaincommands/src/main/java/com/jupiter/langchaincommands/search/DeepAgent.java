package com.jupiter.langchaincommands.search;

import dev.langchain4j.agent.tool.P;
import dev.langchain4j.agent.tool.Tool;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.tool.ToolArgumentsErrorHandler;
import dev.langchain4j.service.tool.ToolExecutionErrorHandler;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Java version of deepagents' create_deep_agent(tools=[internet_search], system_prompt=..., model=llm).
 * Like deepagents, the agent gets a todo-list planning tool and a virtual (in-memory) file system
 * in addition to the user's tools, and loops over tool calls until it writes a final answer.



// Not too much control of the flow then deep agents too much control it is langgraph

        ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┬────────────────────────┬───────────────────────────────────────────────────────────────────────────────────────────┐
        │                                               You want…                                                │          Use           │                                   Who decides the flow                                    │
        ├────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
        │ A lot of control: fixed steps, branching, retries, human-approval checkpoints, predictable paths       │ LangGraph              │ You. You define the nodes, edges and state.                                               │
        ├────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
        │ A simple tool-calling agent                                                                            │ LangChain create_agent │ The model, inside a basic loop that calls tools until it's done                           │
        ├────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
        │ Little control: long, open-ended tasks where the agent plans, breaks work down and runs for many steps │ Deep Agents            │ The model. It writes its own to-do plan, spins up helper agents and saves notes to files. │
        └────────────────────────────────────────────────────────────────────────────────────────────────────────┴────────────────────────┴───────────────────────────────────────────────────────────────────────────────────────────┘
 */
public final class DeepAgent {

    interface Agent {
        String chat(String userMessage);
    }

    /** Built-in instructions deepagents adds on top of the user's system prompt (shortened). */
    private static final String BASE_PROMPT = """

            You have access to these tools:
            - write_todos: plan multi-step work as a todo list and update item statuses as you progress.
            - ls, read_file, write_file, edit_file: a virtual file system for saving notes and reports.
            Work through your todos one by one, then give the user a final answer.""";

    private final Agent agent;
    private final Tools tools;

    /**
     * Builds the agent - the Java version of
     * {@code create_deep_agent(tools=[internet_search], system_prompt=..., model=llm)}.
     * <ul>
     *   <li><b>Tools</b>: internet_search (Tavily) plus the built-in deep-agent tools write_todos, ls, read_file,
     *       write_file and edit_file. LangChain4j reads each {@code @Tool} description and parameter names and sends
     *       them to the model, which decides when to call them.</li>
     *   <li><b>System prompt</b>: the caller's instructions followed by {@link #BASE_PROMPT}, which explains the
     *       built-in tools.</li>
     *   <li><b>Memory</b>: the last 100 messages, so the model sees its earlier tool calls and their results
     *       within one task.</li>
     *   <li><b>Tool errors</b>: invalid tool arguments are explained to the model so it can retry; an exception
     *       inside a tool (e.g. Tavily unreachable) becomes a short generic message, so the agent keeps going and no
     *       internal error details are sent to the LLM.</li>
     * </ul>
     * AiServices runs the agent loop: call the model, run any tools it asks for, send the results back, and repeat
     * until the model replies with plain text.
     *
     * @param model        chat model (gpt-5.4 with strict tool arguments in the demo)
     * @param tavily       web-search client used by internet_search
     * @param systemPrompt the role and working rules, e.g. "You are an expert Researcher, Always (1) ..."
     */
    public DeepAgent(ChatModel model, TavilySearch tavily, String systemPrompt) {
        this.tools = new Tools(tavily);
        this.agent = AiServices.builder(Agent.class)
                .chatModel(model)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(100))
                .systemMessageProvider(memoryId -> systemPrompt + BASE_PROMPT)
                .tools(tools)
                // the model sent invalid tool arguments: tell it what was wrong so it can retry with corrected ones
                .toolArgumentsErrorHandler(ToolArgumentsErrorHandler.sendExceptionMessageToLlm())
                // a tool threw (e.g. a Tavily network error): give the model a generic message so the research can
                // continue, without passing internal error details to the LLM provider
                .toolExecutionErrorHandler(ToolExecutionErrorHandler.sendGenericMessageToLlmUnlessVisibleToLlm(
                        "The tool failed. Try again, or continue with the information you already have."))
                .build();
    }

    /**
     * Gives the agent a task and waits for its final answer - the Java version of
     * {@code web_agent.invoke({"messages": [{"role": "user", "content": ...}]})} followed by reading the last
     * message. While it works, every tool call is printed as {@code [tool] name: details}.
     *
     * @param userMessage the task, e.g. "Latest News in AI Space as of Sep 16th 2026"
     * @return the model's final text answer (the report)
     */
    public String invoke(String userMessage) {
        return agent.chat(userMessage);
    }

    /** Files the agent saved to its virtual file system (path -> content). */
    public Map<String, String> files() {
        return tools.files;
    }

    static final class Tools {

        private final TavilySearch tavily;
        private final Map<String, String> files = new LinkedHashMap<>();
        private List<String> todos = new ArrayList<>();

        Tools(TavilySearch tavily) {
            this.tavily = tavily;
        }

        @Tool("Search the web for current information on topic. Use for any question that needs up-to date "
                + "information, facts or trends or competitor/market information")
        public String internet_search(@P("search query") String query,
                                      @P("maximum number of results; use 5 unless more are needed") int maxResults) {
            log("internet_search", query + " (max " + maxResults + ")");
            return tavily.search(query, maxResults);
        }

        @Tool("Create or replace the todo list used to plan and track a multi-step task. "
                + "Prefix each item with its status, e.g. '[pending] ...', '[in_progress] ...', '[completed] ...'.")
        /** Planning tool: replaces the whole to-do list; the model calls it again to update statuses. */
        public String write_todos(@P("the full todo list") List<String> items) {
            todos = new ArrayList<>(items);
            log("write_todos", String.join(" | ", todos));
            return "Updated todo list to " + todos;
        }

        @Tool("List all files in the virtual file system")
        public String ls() {
            log("ls", "");
            return files.isEmpty() ? "(no files)" : String.join("\n", files.keySet());
        }

        @Tool("Read a file from the virtual file system")
        public String read_file(@P("file path") String path) {
            log("read_file", path);
            String content = files.get(path);
            return content == null ? "Error: file '" + path + "' not found" : content;
        }

        @Tool("Write (create or overwrite) a file in the virtual file system")
        public String write_file(@P("file path") String path, @P("file content") String content) {
            log("write_file", path + " (" + content.length() + " chars)");
            files.put(path, content);
            return "Saved " + path;
        }

        @Tool("Replace an exact string in an existing file of the virtual file system")
        public String edit_file(@P("file path") String path, @P("exact text to replace") String oldString,
                                @P("replacement text") String newString) {
            log("edit_file", path);
            String content = files.get(path);
            if (content == null) {
                return "Error: file '" + path + "' not found";
            }
            if (!content.contains(oldString)) {
                return "Error: text to replace not found in " + path;
            }
            files.put(path, content.replace(oldString, newString));
            return "Edited " + path;
        }

        /** Prints each tool call so you can watch the agent plan, search and save files. */
        private static void log(String tool, String detail) {
            System.out.println("  [tool] " + tool + (detail.isEmpty() ? "" : ": " + detail));
        }
    }
}
