package com.jupiter.langchaincommands.testutil;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/** Offline stand-ins for OpenAI, so the tests need no API key, network or money. */
public final class Fakes {

    private Fakes() {
    }

    /**
     * Fake embeddings: one dimension per known word, value = how often it appears. Texts sharing words get similar
     * vectors, which is enough to test search and retrieval logic.
     */
    public static final class WordEmbeddings implements EmbeddingModel {
        static final List<String> VOCAB = List.of("data", "science", "bepec", "courses", "offers", "learn",
                "langchain", "python", "rag", "founded", "alumni", "headquartered", "machine", "learning", "pizza");

        @Override
        public Response<List<Embedding>> embedAll(List<TextSegment> segments) {
            List<Embedding> out = new ArrayList<>();
            for (TextSegment s : segments) {
                String text = s.text().toLowerCase(Locale.ROOT);
                float[] v = new float[VOCAB.size() + 1];
                for (int i = 0; i < VOCAB.size(); i++) {
                    v[i] = text.split("\\b" + VOCAB.get(i) + "\\b", -1).length - 1;
                }
                v[VOCAB.size()] = 0.01f;
                out.add(Embedding.from(v));
            }
            return Response.from(out);
        }
    }

    /** Fake chat model: returns the scripted replies in order and records every request it receives. */
    public static final class ScriptedChatModel implements ChatModel {
        public final List<ChatRequest> requests = new ArrayList<>();
        private final List<AiMessage> replies;

        public ScriptedChatModel(AiMessage... replies) {
            this.replies = List.of(replies);
        }

        @Override
        public ChatResponse doChat(ChatRequest request) {
            requests.add(request);
            return ChatResponse.builder().aiMessage(replies.get(Math.min(requests.size(), replies.size()) - 1)).build();
        }
    }
}
