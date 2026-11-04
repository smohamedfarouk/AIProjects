package com.jupiter.langchaincommands.chat.chatprompttemplate;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/** Offline test of the prompt-template lesson: a fake model records each prompt, so no OpenAI calls are made. */
class PromptChainTest {

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
    void templateFillsPlaceholdersLikeChatPromptTemplate() {
        var llm = new RecordingLlm();
        PromptChain medical = PromptChain.fromTemplate(Templates.MEDICAL_TEMPLATE, llm);

        assertEquals("reply 1", medical.invoke(Map.of("Name", "Creatine")));
        assertTrue(llm.prompts.getFirst().endsWith("www.bepec.in: Creatine"));
        assertEquals("reply 2", medical.invoke("Paracetamol"), "a plain string fills the template's only placeholder");
        assertTrue(llm.prompts.get(1).endsWith(": Paracetamol"));
        assertThrows(IllegalArgumentException.class, () -> medical.invoke(Map.of("Wrong", "x")));
    }

    @Test
    void draftTemplateTakesTheClientOffer() {
        var llm = new RecordingLlm();
        PromptChain.fromTemplate(Templates.DRAFT_TEMPLATE, llm).invoke(Map.of("input", Templates.CLIENT_OFFER_INPUT_REASON));

        assertTrue(llm.prompts.getFirst().startsWith("Draft a short polite email"));
        assertTrue(llm.prompts.getFirst().endsWith("Low Budget: " + Templates.CLIENT_OFFER_INPUT_REASON));
    }
}
