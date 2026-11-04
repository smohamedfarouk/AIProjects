package com.jupiter.guardrails.common;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.output.TokenUsage;

/** Thin wrapper around the chat model: system prompt + user text -> response text. Each call is an llm span. */
public class LlmClient {

    private final ChatModel model;
    private final String modelName;

    public LlmClient(ChatModel model, String modelName) {
        this.model = model;
        this.modelName = modelName;
    }

    public String ask(String system, String user) {
        return call(system, user);
    }

    /** Single user message, no system prompt (like ChatPromptTemplate.from_template). */
    public String ask(String user) {
        return call(null, user);
    }

    private String call(String system, String user) {
        return Tracing.trace("ChatOpenAI", "llm", span -> {
            span.setAttribute("gen_ai.system", "openai");
            span.setAttribute("gen_ai.request.model", modelName);
            int i = 0;
            if (system != null) {
                span.setAttribute("gen_ai.prompt." + i + ".role", "system");
                span.setAttribute("gen_ai.prompt." + i++ + ".content", system);
            }
            span.setAttribute("gen_ai.prompt." + i + ".role", "user");
            span.setAttribute("gen_ai.prompt." + i + ".content", user);

            ChatResponse response = system != null
                    ? model.chat(SystemMessage.from(system), UserMessage.from(user))
                    : model.chat(UserMessage.from(user));
            String text = response.aiMessage().text();

            span.setAttribute("gen_ai.completion.0.role", "assistant");
            span.setAttribute("gen_ai.completion.0.content", text);
            TokenUsage usage = response.tokenUsage();
            if (usage != null) {
                if (usage.inputTokenCount() != null) span.setAttribute("gen_ai.usage.input_tokens", usage.inputTokenCount());
                if (usage.outputTokenCount() != null) span.setAttribute("gen_ai.usage.output_tokens", usage.outputTokenCount());
            }
            return text;
        });
    }

    public static String content(ChatMessage m) {
        if (m instanceof UserMessage u) return u.singleText();
        if (m instanceof AiMessage a) return a.text();
        if (m instanceof SystemMessage s) return s.text();
        return m.toString();
    }
}
