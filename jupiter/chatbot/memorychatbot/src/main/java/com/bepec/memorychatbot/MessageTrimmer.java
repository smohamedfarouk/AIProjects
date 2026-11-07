package com.bepec.memorychatbot;

import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.TokenCountEstimator;

import java.util.ArrayList;
import java.util.List;

/**
 * Equivalent of {@code trim_messages(max_tokens=1000, strategy="last", include_system=False, start_on="human")}:
 * keeps the most recent messages that fit in the token budget, and drops leading messages until the
 * kept window starts with a user message.
 */
public final class MessageTrimmer {

    private final int maxTokens;
    private final TokenCountEstimator estimator;

    public MessageTrimmer(int maxTokens, TokenCountEstimator estimator) {
        this.maxTokens = maxTokens;
        this.estimator = estimator;
    }

    /**
     * Returns the part of the history that fits in the token budget.
     * <ol>
     *   <li>Walk backwards from the newest message, adding each message's token count.</li>
     *   <li>Stop at the first message that would go over {@code maxTokens}; everything older is left out
     *       (strategy="last": keep the most recent part of the conversation).</li>
     *   <li>Drop messages from the front until the kept part starts with a user message (start_on="human"), so
     *       the model never sees an AI answer without the question it answered.</li>
     * </ol>
     * Example (1 token per word, budget 5): [U "one two three", A "four five", U "six", A "seven eight"] ->
     * the last three fit (2+1+2), then the leading AI message is dropped -> [U "six", A "seven eight"].
     * <p>
     * The input list is not changed; the full history stays in {@link ChatHistory}.
     *
     * @param history all messages of the session, oldest first
     * @return the newest messages within budget, oldest first, starting with a user message (may be empty)
     */
    public List<ChatMessage> trim(List<ChatMessage> history) {
        List<ChatMessage> kept = new ArrayList<>();
        int used = 0;
        for (int i = history.size() - 1; i >= 0; i--) {
            int cost = estimator.estimateTokenCountInMessage(history.get(i));
            if (used + cost > maxTokens) {
                break;
            }
            used += cost;
            kept.addFirst(history.get(i));
        }
        while (!kept.isEmpty() && !(kept.getFirst() instanceof UserMessage)) {
            kept.removeFirst();
        }
        return kept;
    }
}
