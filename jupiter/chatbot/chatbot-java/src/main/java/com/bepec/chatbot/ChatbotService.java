package com.bepec.chatbot;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.TokenCountEstimator;
import dev.langchain4j.model.chat.ChatModel;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

/**
 * Java port of backend.py: few-shot prompt, history trimming,
 * and one-turn conversation with memory.
 */
@Service
public class ChatbotService {

    private static final int HISTORY_MAX_TOKENS = 1000;

    private static final String SYSTEM_PROMPT =
            "You are a helpful, friendly AI assistant built by BEPEC Solutions. "
            + "You remember the ongoing conversation and answer clearly and concisely. "
            + "If you don't know something, say so honestly instead of making things up. "
            + "Match the tone and style of the examples. and end with Thanks for Asking BEPEC";

    /** Few-shot examples, sent as human/ai message pairs (FewShotChatMessagePromptTemplate). */
    private static final List<ChatMessage> FEW_SHOT_EXAMPLES = List.of(
            UserMessage.from("Who are you?"),
            AiMessage.from("I'm the BEPEC assistant. I help you with questions about our "
                    + "AI, Data Science, and Data Engineering courses, placements, and careers."),
            UserMessage.from("I'm from a non-IT background. Can I learn AI?"),
            AiMessage.from("Absolutely. Many BEPEC learners come from non-IT backgrounds. "
                    + "We start from fundamentals and take you to job-ready, step by step. "
                    + "No prior coding needed to begin."),
            UserMessage.from("How long does the course take?"),
            AiMessage.from("It depends on the track and your pace, but most learners become "
                    + "job-ready in a few focused months. Pick a track and commit to "
                    + "consistent daily practice.")
    );

    private final ChatModel llm;
    private final TokenCountEstimator tokenCounter;

    public ChatbotService(ChatModel llm, TokenCountEstimator tokenCounter) {
        this.llm = llm;
        this.tokenCounter = tokenCounter;
    }

    /** Fresh in-memory chat history (demo_memory). One per user session. */
    public List<ChatMessage> newMemory() {
        return new ArrayList<>();
    }

    /**
     * Run one conversation turn (demo_conversation). The user message and reply
     * are appended to {@code memory} only when the call succeeds.
     */
    public String converse(String inputText, List<ChatMessage> memory) {
        synchronized (memory) {
            List<ChatMessage> prompt = new ArrayList<>();
            prompt.add(SystemMessage.from(SYSTEM_PROMPT));
            prompt.addAll(FEW_SHOT_EXAMPLES);
            prompt.addAll(trim(memory));
            UserMessage userMessage = UserMessage.from(inputText);
            prompt.add(userMessage);

            try {
                AiMessage reply = llm.chat(prompt).aiMessage();
                memory.add(userMessage);
                memory.add(reply);
                return reply.text();
            } catch (Exception e) {
                return "Sorry, something went wrong: " + e.getMessage();
            }
        }
    }

    /**
     * Equivalent of trim_messages(max_tokens=1000, strategy="last", start_on="human"):
     * keep the most recent messages that fit the budget, then drop leading
     * messages until the history starts with a user message.
     */
    private List<ChatMessage> trim(List<ChatMessage> history) {
        LinkedList<ChatMessage> kept = new LinkedList<>();
        int total = 0;
        for (int i = history.size() - 1; i >= 0; i--) {
            ChatMessage message = history.get(i);
            int tokens = tokenCounter.estimateTokenCountInMessage(message);
            if (total + tokens > HISTORY_MAX_TOKENS) {
                break;
            }
            total += tokens;
            kept.addFirst(message);
        }
        while (!kept.isEmpty() && !(kept.getFirst() instanceof UserMessage)) {
            kept.removeFirst();
        }
        return kept;
    }
}
