package com.jupiter.guardrails;

import dev.langchain4j.data.message.ChatMessage;

import java.util.ArrayList;
import java.util.List;

/** Shared state passed between graph nodes (mirrors the Python AgentState TypedDict). */
public class AgentState {

    public static final String FINISH = "FINISH";

    private final List<ChatMessage> messages = new ArrayList<>();
    private final String task;
    private String next = "";
    private boolean blocked = false;

    public AgentState(String task) {
        this.task = task;
    }

    public List<ChatMessage> messages() {
        return messages;
    }

    public void addMessage(ChatMessage message) {
        messages.add(message);
    }

    public String task() {
        return task;
    }

    public String next() {
        return next;
    }

    public void setNext(String next) {
        this.next = next;
    }

    public boolean blocked() {
        return blocked;
    }

    public void setBlocked(boolean blocked) {
        this.blocked = blocked;
    }
}
