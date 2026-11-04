package com.jupiter.langchaincommands.rag;

import com.jupiter.langchaincommands.testutil.Fakes.ScriptedChatModel;
import com.jupiter.langchaincommands.testutil.Fakes.WordEmbeddings;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Offline tests of every {@link RagPipeline} step, in pipeline order: sample HTML instead of a web download,
 * word-count embeddings instead of OpenAI, and a scripted chat model that records the prompt.
 */
class RagPipelineTest {

    static final List<String> BEPEC_FACTS = List.of(
            "BEPEC Solutions was founded 13+ years ago in Bangalore.",
            "BEPEC has placed 30,000+ alumni across 500+ hiring partners.",
            "BEPEC Offers this Courses which include Data Analytics, Data Engineering, AI/ML, GenAI.",
            "BEPEC is headquartered in Koramangala, Bengaluru.");

    // 1. LOAD

    @Test
    void fromHtmlKeepsVisibleTextAndMetadata() {
        Document doc = RagPipeline.fromHtml("""
                <html lang="en"><head><title>LangChain overview</title>
                <meta name="description" content="LangChain provides create_agent"></head>
                <body><h1>Hello</h1><script>ignored()</script><p>LangChain helps build LLM apps.</p></body></html>""",
                "https://example.test/docs");

        assertEquals("LangChain overview Hello LangChain helps build LLM apps.", doc.text(), "scripts are dropped");
        assertEquals("https://example.test/docs", doc.metadata().getString("source"));
        assertEquals("LangChain overview", doc.metadata().getString("title"));
        assertEquals("LangChain provides create_agent", doc.metadata().getString("description"));
        assertEquals("en", doc.metadata().getString("language"));
    }

    @Test
    void styleAndScriptInsideTheBodyAreNotText() {
        Document doc = RagPipeline.fromHtml("""
                <!DOCTYPE html><html lang="en"><head><title>Docs</title></head><body>
                <style>.x{display:none}</style><nav>Menu</nav><main><p>Real &amp; visible text</p></main>
                <script>track()</script><noscript>Enable JS</noscript></body></html>""", "https://example.test/");

        assertEquals("Docs Menu Real & visible text", doc.text());
    }

    @Test
    void missingTitleAndDescriptionAreLeftOut() {
        Document doc = RagPipeline.fromHtml("<p>Just text</p>", "https://example.test/");
        assertEquals("Just text", doc.text());
        assertNull(doc.metadata().getString("title"));
    }

    // 2. CHUNK

    @Test
    void chunksRespectTheSizeLimitAndKeepMetadata() {
        String text = "LangChain is a framework for building applications powered by language models. ".repeat(20);

        List<String> chunks = RagPipeline.split(text, 200, 30);
        assertTrue(chunks.size() > 5, "1,600 characters should give many chunks, got " + chunks.size());
        chunks.forEach(c -> assertTrue(c.length() <= 200, "chunk too long: " + c.length()));

        Document page = RagPipeline.fromHtml("<title>Intro</title><p>" + text + "</p>", "https://example.test/intro");
        List<TextSegment> segments = RagPipeline.splitDocument(page, 200, 30);
        assertTrue(segments.stream().allMatch(s -> "https://example.test/intro".equals(s.metadata().getString("source"))),
                "every chunk remembers where it came from");
    }

    // 3. EMBED

    @Test
    void similarTextsHaveHigherCosineSimilarity() {
        var embeddings = new WordEmbeddings();
        Embedding question = RagPipeline.embedQuery(embeddings, "What is machine learning?");
        List<Embedding> docs = RagPipeline.embedDocuments(embeddings,
                List.of("Machine learning is a subset of AI.", "Pizza is a popular Italian dish."));

        assertEquals(2, docs.size());
        assertTrue(RagPipeline.cosine(question, docs.get(0)) > RagPipeline.cosine(question, docs.get(1)));
        assertEquals(1.0, RagPipeline.cosine(question, question), 1e-6);
    }

    // 4. STORE + SEARCH

    @Test
    void searchReturnsTheClosestDocumentsFirst() {
        var embeddings = new WordEmbeddings();
        var store = RagPipeline.fromDocuments(List.of(
                "BEPEC offers Data Science courses in Bangalore.",
                "LangChain helps build LLM applications quickly.",
                "Python is the most popular language for AI.",
                "RAG combines retrieval with text generation in Data Science"), embeddings);

        List<EmbeddingMatch<TextSegment>> top2 =
                RagPipeline.similaritySearch(store, embeddings, "where can i learn data science", 2);
        List<String> texts = top2.stream().map(m -> m.embedded().text()).toList();

        assertTrue(texts.contains("BEPEC offers Data Science courses in Bangalore."), texts.toString());
        assertTrue(texts.contains("RAG combines retrieval with text generation in Data Science"), texts.toString());
        assertTrue(top2.get(0).score() >= top2.get(1).score(), "most similar first");
    }

    // 5. ANSWER

    @Test
    void answerPutsTheRetrievedFactsAndTheQuestionIntoThePrompt() {
        var embeddings = new WordEmbeddings();
        var store = RagPipeline.fromDocuments(BEPEC_FACTS, embeddings);
        var llm = new ScriptedChatModel(AiMessage.from("BEPEC offers Data Analytics, Data Engineering, AI/ML, GenAI."));

        String answer = RagPipeline.answer(store, embeddings, llm, "What courses BEPEC offers", 2);

        assertEquals("BEPEC offers Data Analytics, Data Engineering, AI/ML, GenAI.", answer);
        String prompt = ((UserMessage) llm.requests.getFirst().messages().getLast()).singleText();
        assertTrue(prompt.contains("Answer the question based only on the context below."));
        assertTrue(prompt.contains("BEPEC Offers this Courses which include Data Analytics"), prompt);
        assertTrue(prompt.contains("Question: What courses BEPEC offers"));
        // the question mentions BEPEC once; each retrieved fact once more -> exactly k = 2 facts in the context
        assertEquals(3, prompt.split("BEPEC", -1).length - 1, prompt);
    }
}
