package com.bepec.memorychatbot;

import dev.langchain4j.data.message.ChatMessage;

import java.util.ArrayList;
import java.util.List;

/**
 * Equivalent of InMemoryChatMessageHistory: the full, untrimmed conversation for one session.
 * Trimming happens only when the prompt is built, so nothing is ever lost from the history.
 */
public final class ChatHistory {

    private final List<ChatMessage> messages = new ArrayList<>();

    /** A read-only snapshot of all messages, oldest first. A copy, so callers cannot change the history by accident. */
    public synchronized List<ChatMessage> messages() {
        return List.copyOf(messages);
    }

    /** Appends one message (the user's question or the AI's reply) to the end of the conversation. */
    public synchronized void add(ChatMessage message) {
        messages.add(message);
    }

    /** Forgets the whole conversation - used by the "Clear Conversation" button. */
    public synchronized void clear() {
        messages.clear();
    }
}
