package com.bepec.deepagents.agent;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;

/** Java stand-in for LangChain's {@code message.pretty_print()}. */
public final class MessagePrinter {

    private MessagePrinter() {
    }

    public static void prettyPrint(ChatMessage message) {
        System.out.println(format(message));
    }

    public static String format(ChatMessage message) {
        StringBuilder sb = new StringBuilder();
        switch (message) {
            case SystemMessage m -> sb.append(header("System Message")).append(m.text());
            case UserMessage m -> sb.append(header("Human Message")).append(m.singleText());
            case AiMessage m -> {
                sb.append(header("Ai Message"));
                if (m.text() != null && !m.text().isBlank()) {
                    sb.append(m.text()).append('\n');
                }
                if (m.hasToolExecutionRequests()) {
                    sb.append("Tool Calls:\n");
                    for (ToolExecutionRequest r : m.toolExecutionRequests()) {
                        sb.append("  ").append(r.name()).append(" (").append(r.id()).append(")\n")
                          .append("    Args: ").append(r.arguments()).append('\n');
                    }
                }
            }
            case ToolExecutionResultMessage m -> sb.append(header("Tool Message"))
                    .append("Name: ").append(m.toolName()).append("\n\n").append(m.text());
            default -> sb.append(message);
        }
        return sb.toString();
    }

    private static String header(String title) {
        String bar = "=".repeat(32);
        return bar + " " + title + " " + bar + "\n\n";
    }
}
