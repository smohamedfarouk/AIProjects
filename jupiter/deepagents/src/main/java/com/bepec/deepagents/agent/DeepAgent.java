package com.bepec.deepagents.agent;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import dev.langchain4j.agent.tool.Tool;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.agent.tool.ToolSpecifications;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import dev.langchain4j.service.tool.DefaultToolExecutor;
import dev.langchain4j.service.tool.ToolExecutor;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Consumer;

/**
 * Minimal Java equivalent of Python's {@code deepagents.create_deep_agent(...)}.
 * <p>
 * Every agent gets planning ({@code write_todos}) and a virtual file system
 * ({@code ls/read_file/write_file/edit_file}). When subagents are configured the agent also gets a
 * {@code task} tool that runs a subagent in a fresh context sharing the same file system.
 */
public final class DeepAgent {

    static final String BASE_PROMPT = """

            You have access to a `write_todos` tool to plan and track multi-step work, and to a virtual \
            file system (`ls`, `read_file`, `write_file`, `edit_file`) to store notes and results.""";

    static final String TASK_PROMPT = """

            Use the `task` tool to delegate work to a specialised subagent. Each subagent starts with a \
            fresh context, shares your file system, and returns only its final answer.""";

    private static final ObjectMapper JSON = new ObjectMapper();
    private static final int MAX_STEPS = 50;

    private final ChatModel model;
    private final String systemPrompt;
    private final List<Object> tools;
    private final Map<String, SubAgent> subagents;

    private DeepAgent(Builder b) {
        this.model = Objects.requireNonNull(b.model, "model is required");
        this.systemPrompt = b.systemPrompt == null ? "" : b.systemPrompt;
        this.tools = List.copyOf(b.tools);
        this.subagents = new LinkedHashMap<>();
        b.subagents.forEach(s -> subagents.put(s.name(), s));
    }

    public static Builder builder() {
        return new Builder();
    }

    /** {@code agent.invoke({"messages": [{"role": "user", "content": input}]})}. */
    public AgentResult invoke(String input) {
        return stream(input, m -> { });
    }

    /** {@code agent.stream(..., stream_mode="values")}: every new message is passed to {@code onMessage}. */
    public AgentResult stream(String input, Consumer<ChatMessage> onMessage) {
        VirtualFileSystemTools fs = new VirtualFileSystemTools();
        List<ChatMessage> messages = run(systemPrompt, tools, true, fs, input, onMessage);
        return new AgentResult(messages, fs.files());
    }

    private List<ChatMessage> run(String prompt, List<Object> userTools, boolean withTask,
                                  VirtualFileSystemTools fs, String input, Consumer<ChatMessage> onMessage) {
        List<Object> all = new ArrayList<>(userTools);
        all.add(new TodoTools());
        all.add(fs);

        List<ToolSpecification> specs = new ArrayList<>();
        Map<String, ToolExecutor> executors = new LinkedHashMap<>();
        for (Object tool : all) {
            for (Method method : tool.getClass().getDeclaredMethods()) {
                if (method.isAnnotationPresent(Tool.class)) {
                    ToolSpecification spec = ToolSpecifications.toolSpecificationFrom(method);
                    if (executors.putIfAbsent(spec.name(), new DefaultToolExecutor(tool, method)) == null) {
                        specs.add(spec);
                    }
                }
            }
        }
        boolean task = withTask && !subagents.isEmpty();
        if (task) {
            specs.add(taskSpecification());
        }

        List<ChatMessage> messages = new ArrayList<>();
        add(messages, SystemMessage.from(prompt + BASE_PROMPT + (task ? TASK_PROMPT : "")), onMessage);
        add(messages, UserMessage.from(input), onMessage);

        for (int step = 0; step < MAX_STEPS; step++) {
            ChatRequest request = ChatRequest.builder().messages(messages).toolSpecifications(specs).build();
            AiMessage ai = model.chat(request).aiMessage();
            add(messages, ai, onMessage);
            if (!ai.hasToolExecutionRequests()) {
                return messages;
            }
            for (ToolExecutionRequest call : ai.toolExecutionRequests()) {
                String result;
                try {
                    if (task && "task".equals(call.name())) {
                        result = runTask(call.arguments(), fs, onMessage);
                    } else {
                        ToolExecutor executor = executors.get(call.name());
                        result = executor == null
                                ? "Error: unknown tool '" + call.name() + "'"
                                : executor.execute(call, null);
                    }
                } catch (Exception e) {
                    result = "Error: " + e.getMessage();
                }
                add(messages, ToolExecutionResultMessage.from(call, result), onMessage);
            }
        }
        throw new IllegalStateException("Agent exceeded " + MAX_STEPS + " steps without a final answer");
    }

    private String runTask(String arguments, VirtualFileSystemTools fs, Consumer<ChatMessage> onMessage)
            throws Exception {
        JsonNode args = JSON.readTree(arguments);
        String type = args.path("subagent_type").asText();
        SubAgent sub = subagents.get(type);
        if (sub == null) {
            sub = subagents.get(type.replace('_', '-'));
        }
        if (sub == null) {
            return "Error: unknown subagent '" + type + "'. Available: " + subagents.keySet();
        }
        List<ChatMessage> subMessages = run(sub.systemPrompt(), sub.tools(), false, fs,
                args.path("description").asText(), onMessage);
        return new AgentResult(subMessages, Map.of()).lastText();
    }

    private ToolSpecification taskSpecification() {
        StringBuilder desc = new StringBuilder("Launch a subagent to handle a task. Available subagents:\n");
        subagents.values().forEach(s -> desc.append("- ").append(s.name()).append(": ")
                .append(s.description()).append('\n'));
        return ToolSpecification.builder()
                .name("task")
                .description(desc.toString())
                .parameters(JsonObjectSchema.builder()
                        .addStringProperty("description", "detailed description of the task for the subagent")
                        .addEnumProperty("subagent_type", List.copyOf(subagents.keySet()),
                                "which subagent to use")
                        .required("description", "subagent_type")
                        .build())
                .build();
    }

    private static void add(List<ChatMessage> messages, ChatMessage message, Consumer<ChatMessage> onMessage) {
        messages.add(message);
        onMessage.accept(message);
    }

    public static final class Builder {
        private ChatModel model;
        private String systemPrompt;
        private final List<Object> tools = new ArrayList<>();
        private final List<SubAgent> subagents = new ArrayList<>();

        public Builder model(ChatModel model) {
            this.model = model;
            return this;
        }

        public Builder systemPrompt(String systemPrompt) {
            this.systemPrompt = systemPrompt;
            return this;
        }

        public Builder tools(Object... tools) {
            this.tools.addAll(Arrays.asList(tools));
            return this;
        }

        public Builder subagents(SubAgent... subagents) {
            this.subagents.addAll(Arrays.asList(subagents));
            return this;
        }

        public DeepAgent build() {
            return new DeepAgent(this);
        }
    }
}
