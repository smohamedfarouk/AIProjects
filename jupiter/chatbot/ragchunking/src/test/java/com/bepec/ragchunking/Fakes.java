package com.bepec.ragchunking;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.function.Function;

/** Offline stand-ins for OpenAI, so the tests are free and need no API key. */
final class Fakes {

    private Fakes() {
    }

    /** Bag-of-words embeddings: one dimension per vocabulary word. Texts sharing words get similar vectors. */
    static final class WordEmbeddings implements EmbeddingModel {
        static final List<String> VOCAB = List.of("machine", "learning", "data", "deep", "neural", "networks",
                "layers", "rag", "retrieval", "generation", "hallucinations", "langgraph", "workflows", "vector",
                "embeddings", "search");

        @Override
        public Response<List<Embedding>> embedAll(List<TextSegment> segments) {
            List<Embedding> out = new ArrayList<>();
            for (TextSegment s : segments) {
                String text = s.text().toLowerCase(Locale.ROOT);
                float[] v = new float[VOCAB.size() + 1];
                for (int i = 0; i < VOCAB.size(); i++) {
                    v[i] = text.split("\\b" + VOCAB.get(i) + "\\b", -1).length - 1;
                }
                v[VOCAB.size()] = 0.01f; // never an all-zero vector
                out.add(Embedding.from(v));
            }
            return Response.from(out);
        }
    }

    /** Fake LLM: records every request and answers with whatever the function returns. */
    static final class FakeLlm implements ChatModel {
        final List<ChatRequest> requests = new ArrayList<>();
        private final Function<ChatRequest, AiMessage> reply;

        FakeLlm(Function<ChatRequest, AiMessage> reply) {
            this.reply = reply;
        }

        static FakeLlm answering(String text) {
            return new FakeLlm(r -> AiMessage.from(text));
        }

        @Override
        public ChatResponse doChat(ChatRequest request) {
            requests.add(request);
            return ChatResponse.builder().aiMessage(reply.apply(request)).build();
        }

        /** Text of the last message of request i. */
        String prompt(int i) {
            ChatMessage last = requests.get(i).messages().getLast();
            return ((dev.langchain4j.data.message.UserMessage) last).singleText();
        }
    }
}
