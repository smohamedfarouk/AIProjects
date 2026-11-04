package com.jupiter.langchaincommands.chat.selfcorrectingprompt;

import com.jupiter.langchaincommands.chat.chatprompttemplate.Templates;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/** Offline test of the reflection lesson: a fake model records each prompt, so no OpenAI calls are made. */
class ReflectionChainTest {

    /** Replies "reply N" and remembers every prompt text it received. */
    static final class RecordingLlm implements ChatModel {
        final List<String> prompts = new ArrayList<>();

        @Override
        public ChatResponse doChat(ChatRequest request) {
            prompts.add(((UserMessage) request.messages().getLast()).singleText());
            return ChatResponse.builder().aiMessage(AiMessage.from("reply " + prompts.size())).build();
        }
    }

    @Test
    void reflectionRunsDraftThenCritiqueThenRefine() {
        var llm = new RecordingLlm();
        String result = new ReflectionChain(llm).invoke(Templates.CLIENT_OFFER_INPUT_REASON);

        assertEquals("reply 3", result, "only the refined email is returned");
        assertEquals(3, llm.prompts.size());
        assertTrue(llm.prompts.get(0).startsWith("Draft a short polite email") && llm.prompts.get(0).contains("25000/day"));
        assertTrue(llm.prompts.get(1).contains("Draft: reply 1"), "critique sees the draft");
        assertTrue(llm.prompts.get(2).contains("Original Draft: reply 1") && llm.prompts.get(2).contains("Critique: reply 2"),
                "refine sees the draft and the critique");
    }
}
