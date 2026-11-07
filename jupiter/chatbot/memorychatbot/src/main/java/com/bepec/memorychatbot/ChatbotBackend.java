package com.bepec.memorychatbot;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.response.ChatResponse;

import java.util.ArrayList;
import java.util.List;

/**
 * Java port of backend.py. Each turn sends:
 * system prompt -> few-shot examples -> trimmed history -> new user input,
 * then stores the user input and the reply in the session's {@link ChatHistory}.
 */
public final class ChatbotBackend {

    static final String SYSTEM_PROMPT =
            "You are a helpful, friendly AI assistant built by BEPEC Solutions. "
                    + "You remember the ongoing conversation and answer clearly and concisely. "
                    + "If you don't know something, say so honestly instead of making things up. "
                    + "Match the tone and style of the examples. and end with Thanks for Asking BEPEC";

    record Example(String input, String output) {
    }

    static final List<Example> EXAMPLES = List.of(
            new Example("Who are you?",
                    "I'm the BEPEC assistant. I help you with questions about our "
                            + "AI, Data Science, and Data Engineering courses, placements, and careers."),
            new Example("I'm from a non-IT background. Can I learn AI?",
                    "Absolutely. Many BEPEC learners come from non-IT backgrounds. "
                            + "We start from fundamentals and take you to job-ready, step by step. "
                            + "No prior coding needed to begin."),
            new Example("How long does the course take?",
                    "It depends on the track and your pace, but most learners become "
                            + "job-ready in a few focused months. Pick a track and commit to "
                            + "consistent daily practice."));

    private final ChatModel llm;
    private final MessageTrimmer trimmer;

    public ChatbotBackend(ChatModel llm, MessageTrimmer trimmer) {
        this.llm = llm;
        this.trimmer = trimmer;
    }

    /** Equivalent of demo_memory(): a fresh history for a new session. */
    public static ChatHistory newMemory() {
        return new ChatHistory();
    }

    /**
     * Builds the full prompt for one turn - the Java version of CHAT_PROMPT plus the history step of the chain.
     * The order matters:
     * <ol>
     *   <li><b>System message</b> - who the bot is and how to answer (always first).</li>
     *   <li><b>Few-shot examples</b> - each example as a fake user question followed by the ideal AI answer, so the
     *       model copies their tone and length (FewShotChatMessagePromptTemplate).</li>
     *   <li><b>History</b> - this session's earlier messages, cut to the newest 1000 tokens by
     *       {@link MessageTrimmer} (MessagesPlaceholder("history") + trim_messages).</li>
     *   <li><b>New user message</b> - always last, so the model answers it.</li>
     * </ol>
     * Package-private so tests can check the order without calling OpenAI.
     */
    List<ChatMessage> buildPrompt(String input, ChatHistory memory) {
        List<ChatMessage> prompt = new ArrayList<>();
        prompt.add(SystemMessage.from(SYSTEM_PROMPT));
        for (Example example : EXAMPLES) {
            prompt.add(UserMessage.from(example.input()));
            prompt.add(AiMessage.from(example.output()));
        }
        prompt.addAll(trimmer.trim(memory.messages()));
        prompt.add(UserMessage.from(input));
        return prompt;
    }

    /**
     * Runs one conversation turn - the Java version of {@code demo_conversation(input_text, memory)}.
     * <ol>
     *   <li>Build the prompt with {@link #buildPrompt} (history is read, not changed).</li>
     *   <li>Send it to the model and take the reply text.</li>
     *   <li><b>Only after a successful reply</b>, append the user message and the AI reply to {@code memory}
     *       (what RunnableWithMessageHistory does automatically in Python). Next turn they become history.</li>
     *   <li>On any error (no network, bad key, rate limit) return "Sorry, something went wrong: ..." and save
     *       nothing, so a failed turn does not leave a question without an answer in the history.</li>
     * </ol>
     *
     * @param input  what the user typed
     * @param memory this session's history (from {@link #newMemory()})
     * @return the assistant's reply, or a friendly error message
     */
    public String converse(String input, ChatHistory memory) {
        try {
            ChatResponse response = llm.chat(buildPrompt(input, memory));
            String reply = response.aiMessage().text();
            memory.add(UserMessage.from(input));
            memory.add(AiMessage.from(reply));
            return reply;
        } catch (Exception e) {
            return "Sorry, something went wrong: " + e.getMessage();
        }
    }
}
