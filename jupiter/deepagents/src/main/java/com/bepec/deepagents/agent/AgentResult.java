package com.bepec.deepagents.agent;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;

import java.util.List;
import java.util.Map;

/**
 * Result of {@link DeepAgent#invoke(String)}: the Python {@code {"messages": [...], "files": {...}}} state.
 *
 * @param messages full conversation, including the system message and tool calls/results
 * @param files    snapshot of the virtual file system after the run
 */
public record AgentResult(List<ChatMessage> messages, Map<String, String> files) {

    /** Equivalent of {@code out["messages"][-1].content}. */
    public String lastText() {
        for (int i = messages.size() - 1; i >= 0; i--) {
            if (messages.get(i) instanceof AiMessage ai && ai.text() != null) {
                return ai.text();
            }
        }
        return "";
    }
}
