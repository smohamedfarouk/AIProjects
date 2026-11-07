package com.bepec.memorychatbot;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.TokenCountEstimator;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/** Offline tests: a fake LLM and a fake token counter, so no OpenAI calls (free to run). */
class ChatbotBackendTest {

    /** Counts 1 token per word. */
    static final TokenCountEstimator WORDS = new TokenCountEstimator() {
        public int estimateTokenCountInText(String text) {
            return text.isBlank() ? 0 : text.trim().split("\\s+").length;
        }

        public int estimateTokenCountInMessage(ChatMessage m) {
            return estimateTokenCountInText(m instanceof UserMessage u ? u.singleText() : ((AiMessage) m).text());
        }

        public int estimateTokenCountInMessages(Iterable<ChatMessage> ms) {
            int n = 0;
            for (ChatMessage m : ms) n += estimateTokenCountInMessage(m);
            return n;
        }
    };

    /** Records every request and replies with a fixed answer. */
    static final class FakeLlm implements ChatModel {
        final List<List<ChatMessage>> requests = new ArrayList<>();

        @Override
        public ChatResponse doChat(ChatRequest request) {
            requests.add(request.messages());
            return ChatResponse.builder().aiMessage(AiMessage.from("reply " + requests.size())).build();
        }
    }

    @Test
    void trimmerKeepsLatestMessagesWithinBudgetAndStartsOnUser() {
        List<ChatMessage> history = List.of(
                UserMessage.from("one two three"),
                AiMessage.from("four five"),
                UserMessage.from("six"),
                AiMessage.from("seven eight"));

        // budget 4: only [six, seven eight] fit (1 + 2 tokens)
        assertEquals(List.of(history.get(2), history.get(3)), new MessageTrimmer(4, WORDS).trim(history));

        // budget 5: [four five, six, seven eight] fit, but the window must start on a user message
        assertEquals(List.of(history.get(2), history.get(3)), new MessageTrimmer(5, WORDS).trim(history));
        assertEquals(history, new MessageTrimmer(100, WORDS).trim(history));
        assertTrue(new MessageTrimmer(1, WORDS).trim(history).isEmpty());
    }

    @Test
    void promptHasSystemThenFewShotThenHistoryThenInput() {
        var backend = new ChatbotBackend(new FakeLlm(), new MessageTrimmer(1000, WORDS));
        ChatHistory memory = ChatbotBackend.newMemory();
        memory.add(UserMessage.from("My name is Asha"));
        memory.add(AiMessage.from("Hi Asha"));

        List<ChatMessage> prompt = backend.buildPrompt("What is my name?", memory);

        assertInstanceOf(SystemMessage.class, prompt.getFirst());
        assertEquals(1 + 2 * ChatbotBackend.EXAMPLES.size() + 2 + 1, prompt.size());
        assertEquals("Who are you?", ((UserMessage) prompt.get(1)).singleText());
        assertEquals("My name is Asha", ((UserMessage) prompt.get(prompt.size() - 3)).singleText());
        assertEquals("What is my name?", ((UserMessage) prompt.getLast()).singleText());
    }

    @Test
    void converseRemembersEarlierTurns() {
        var llm = new FakeLlm();
        var backend = new ChatbotBackend(llm, new MessageTrimmer(1000, WORDS));
        ChatHistory memory = ChatbotBackend.newMemory();

        assertEquals("reply 1", backend.converse("My name is Asha", memory));
        assertEquals("reply 2", backend.converse("What is my name?", memory));

        assertEquals(4, memory.messages().size());
        List<ChatMessage> second = llm.requests.get(1);
        assertTrue(second.contains(UserMessage.from("My name is Asha")));
        assertTrue(second.contains(AiMessage.from("reply 1")));
    }

    @Test
    void errorsBecomeAFriendlyReplyAndAreNotStored() {
        ChatModel failing = new ChatModel() {
            @Override
            public ChatResponse doChat(ChatRequest request) {
                throw new RuntimeException("boom");
            }
        };
        ChatHistory memory = ChatbotBackend.newMemory();
        String reply = new ChatbotBackend(failing, new MessageTrimmer(1000, WORDS)).converse("hi", memory);

        assertEquals("Sorry, something went wrong: boom", reply);
        assertTrue(memory.messages().isEmpty());
    }
}
