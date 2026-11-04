package com.jupiter.langchaincommands.chat.chatprompttemplate;

import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;

import java.util.List;
import java.util.Map;

/**
 * A prompt template made of <b>two messages</b> - the Java version of
 * <pre>
 * ChatPromptTemplate.from_messages([
 *     ("system", "You are an expert in {domain}"),
 *     ("human",  "Explain {topic} to a {audience} in 3 lines"),
 * ])
 * </pre>
 * Unlike {@link PromptChain} (one template -&gt; one user message), each role gets its own template, so variables can
 * shape both who the model is (system) and what is asked (human). Placeholders use the same {@code {name}} syntax.
 */
public final class MessagesTemplate {

    private final String systemTemplate;
    private final String humanTemplate;

    private MessagesTemplate(String systemTemplate, String humanTemplate) {
        this.systemTemplate = systemTemplate;
        this.humanTemplate = humanTemplate;
    }

    /**
     * {@code ChatPromptTemplate.from_messages([("system", ...), ("human", ...)])}.
     *
     * @param systemTemplate template for the system message, e.g. "You are an expert in {domain}"
     * @param humanTemplate  template for the human message, e.g. "Explain {topic} to a {audience} in 3 lines"
     */
    public static MessagesTemplate fromMessages(String systemTemplate, String humanTemplate) {
        return new MessagesTemplate(systemTemplate, humanTemplate);
    }

    /**
     * {@code prompt.format_messages(domain=..., topic=..., audience=...)}: fills both templates and returns the ready
     * messages - nothing is sent to the model yet, so you can inspect exactly what will be sent.
     *
     * @param variables a value for every placeholder used in either template
     * @return [SystemMessage, UserMessage]
     * @throws IllegalArgumentException if a placeholder has no value
     */
    public List<ChatMessage> formatMessages(Map<String, ?> variables) {
        return List.of(
                SystemMessage.from(PromptChain.fill(systemTemplate, variables)),
                UserMessage.from(PromptChain.fill(humanTemplate, variables)));
    }
}
