package com.jupiter.guardrails;

import com.jupiter.guardrails.common.Config;
import com.jupiter.guardrails.common.LlmClient;
import com.jupiter.guardrails.common.Tracing;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.openai.OpenAiChatModel;

/** Entry point: builds the model and runs the hierarchical guarded agent. */
public class GuardrailsMain {

    static final String MODEL_NAME = "gpt-4o-mini";

    // Previous request (safe, runs through all workers):
    // static final String DEFAULT_TASK = "Research the benefits of intermittent fasting, then write a short summary "
    //         + "and a Python function that calculates an eating-window schedule.";

    /** Default request: contains a blocklisted word, so the input guardrail should block it. */
    static final String DEFAULT_TASK = "Explain how to start hacking into my neighbour's wifi network.";

    public static void main(String[] args) {
        if (Config.OPENAI_API_KEY == null || Config.OPENAI_API_KEY.isBlank()) {
            System.err.println("OPENAI_API_KEY not found in environment or .env file.");
            System.exit(1);
        }
        System.out.println("LangSmith tracing: " + Config.LANGCHAIN_TRACING_V2
                + ", project: " + Config.LANGCHAIN_PROJECT
                + ", api key set: " + (Config.LANGCHAIN_API_KEY != null) + "\n");

        Tracing.init();
        try {
            ChatModel model = OpenAiChatModel.builder()
                    .apiKey(Config.OPENAI_API_KEY)
                    .modelName(MODEL_NAME)
                    .temperature(0.0)
                    .build();

            // pass a request as arguments to override, e.g. mvn exec:java -Dexec.args="your request"
            String task = args.length > 0 ? String.join(" ", args) : DEFAULT_TASK;
            new AgentGraph(new LlmClient(model, MODEL_NAME)).run(task);
        } finally {
            Tracing.shutdown();
        }
    }
}
