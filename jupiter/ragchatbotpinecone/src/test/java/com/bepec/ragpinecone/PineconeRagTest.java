package com.bepec.ragpinecone;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import static dev.langchain4j.store.embedding.filter.MetadataFilterBuilder.metadataKey;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Offline tests: an in-memory store stands in for Pinecone and fake models stand in for OpenAI, so no API keys,
 * network or cloud resources are used.
 */
class PineconeRagTest {

    /** Fake embeddings: one dimension per known word (word counts), so texts sharing words are "similar". */
    static final class WordEmbeddings implements EmbeddingModel {
        static final List<String> VOCAB = List.of("pinecone", "vector", "database", "langchain", "python",
                "loops", "sql", "joins", "internship", "rag", "agentic");

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

    static final class FakeLlm implements ChatModel {
        List<ChatMessage> lastPrompt;

        @Override
        public ChatResponse doChat(ChatRequest request) {
            lastPrompt = request.messages();
            return ChatResponse.builder().aiMessage(AiMessage.from("from context")).build();
        }
    }

    static Document page(int page, String text) {
        return Document.from(text, new Metadata().put("source", "agenda.pdf").put("page", page));
    }

    @Test
    void chunkIdsMatchThePythonNotebook() {
        TextSegment chunk = TextSegment.from("PROGRAM HIGHLIGHTS – 7 Months",
                Metadata.from("source", "AI Agenda_Updated'26.pdf"));
        // hashlib.md5("AI Agenda_Updated'26.pdf::PROGRAM HIGHLIGHTS – 7 Months".encode()).hexdigest()
        assertEquals("acfec4b62c038346663cd3dc5cce03ec", PineconeRagPipeline.chunkId(chunk));
    }

    @Test
    void chunkIdsAreStableAndUniquePerChunk() {
        StringBuilder text = new StringBuilder();
        for (int week = 1; week <= 50; week++) {
            text.append("Week ").append(week).append(": Python loops and lists are practised with real examples. ");
        }
        List<Document> pages = List.of(page(0, text.toString()), page(1, "SQL joins."));
        List<TextSegment> chunks = PineconeRagPipeline.split(pages);

        assertTrue(chunks.size() > 3);
        List<String> ids = chunks.stream().map(PineconeRagPipeline::chunkId).toList();
        assertEquals(ids.size(), new HashSet<>(ids).size(), "different chunk text -> different id");
        assertEquals(ids, PineconeRagPipeline.split(pages).stream().map(PineconeRagPipeline::chunkId).toList(),
                "re-running gives the same ids, so Pinecone overwrites instead of duplicating");
        chunks.forEach(c -> assertTrue(c.text().length() <= PineconeRagPipeline.CHUNK_SIZE));
    }

    @Test
    void identicalTextInTheSameFileSharesOneId() {
        // same formula as the notebook: a header repeated word-for-word on two pages is stored once
        TextSegment onPage2 = TextSegment.from("BEPEC SOLUTIONS", new Metadata().put("source", "a.pdf").put("page", 1));
        TextSegment onPage9 = TextSegment.from("BEPEC SOLUTIONS", new Metadata().put("source", "a.pdf").put("page", 8));
        TextSegment otherFile = TextSegment.from("BEPEC SOLUTIONS", Metadata.from("source", "b.pdf"));

        assertEquals(PineconeRagPipeline.chunkId(onPage2), PineconeRagPipeline.chunkId(onPage9));
        assertNotEquals(PineconeRagPipeline.chunkId(onPage2), PineconeRagPipeline.chunkId(otherFile));
    }

    @Test
    void indexStoresUnderHashIdsAndAnswerUsesTop3WithPageLabels() {
        var store = new InMemoryEmbeddingStore<TextSegment>();
        var llm = new FakeLlm();
        var rag = new PineconeRagPipeline(new WordEmbeddings(), store, llm);

        List<Document> pages = List.of(
                page(0, "Python basics: loops and lists."),
                page(1, "SQL joins on tables."),
                page(2, "3 months internship."),
                page(3, "More Python loops practice."));
        assertEquals(4, rag.index(pages));

        List<EmbeddingMatch<TextSegment>> matches = rag.retrieve("python loops");
        assertEquals(PineconeRagPipeline.TOP_K, matches.size());
        assertEquals(PineconeRagPipeline.chunkId(matches.getFirst().embedded()), matches.getFirst().embeddingId());

        assertEquals("from context", rag.answer("What is covered in Python loops?"));
        String system = ((SystemMessage) llm.lastPrompt.getFirst()).text();
        assertTrue(system.contains("Answer the question using ONLY the context below"));
        assertTrue(system.contains("[Page 1]\nPython basics") && system.contains("[Page 4]\nMore Python"), system);
        assertFalse(system.contains("SQL joins"), "a 4th, unrelated page must not be in the top 3: " + system);
    }

    @Test
    void pageNumbersStoredAsDecimalsStillFormatCorrectly() {
        // Pinecone returns numbers as decimals, e.g. page 4.0 written by the Python notebook
        TextSegment fromPinecone = TextSegment.from("Mock interviews", Metadata.from(Map.of("page", 4.0)));
        assertEquals("[Page 5]\nMock interviews", PineconeRagPipeline.formatDocs(List.of(fromPinecone)));
        assertEquals("[Page 1]\nno page", PineconeRagPipeline.formatDocs(List.of(TextSegment.from("no page"))));
    }

    @Test
    void basicsDemoAddsSearchesAndFiltersByMetadata() {
        var store = new InMemoryEmbeddingStore<TextSegment>();
        var embeddings = new WordEmbeddings();

        assertEquals(List.of("doc1", "doc2"), PineconeBasicsMain.addDocuments(embeddings, store, List.of("doc1", "doc2"),
                List.of(TextSegment.from("Pinecone is a managed vector database.", Metadata.from("source", "notes")),
                        TextSegment.from("LangChain helps build LLM applications.", Metadata.from("source", "notes")))));
        PineconeBasicsMain.addDocuments(embeddings, store, List.of("doc3", "doc4"),
                List.of(TextSegment.from("Agentic AI systems plan and use tools.", Metadata.from("topic", "agents")),
                        TextSegment.from("RAG combines retrieval with generation.", Metadata.from("topic", "rag"))));

        var top = PineconeBasicsMain.similaritySearch(embeddings, store, "What is Pinecone?", 2, null);
        assertEquals("doc1", top.getFirst().embeddingId());

        var filtered = PineconeBasicsMain.similaritySearch(embeddings, store, "rag vector database", 4,
                metadataKey("source").isEqualTo("notes"));
        assertEquals(2, filtered.size(), "filter keeps only source = notes");
        assertTrue(filtered.stream().allMatch(m -> "notes".equals(m.embedded().metadata().getString("source"))));
        assertTrue(filtered.getFirst().score() >= filtered.get(1).score());
    }

    @Test
    void emptyPdfIsRejected() {
        var rag = new PineconeRagPipeline(new WordEmbeddings(), new InMemoryEmbeddingStore<>(), new FakeLlm());
        assertThrows(IllegalArgumentException.class, () -> rag.index(List.of(page(0, "   "))));
    }
}
