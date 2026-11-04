package com.bepec.ragchatbot;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDPage;
import org.apache.pdfbox.pdmodel.PDPageContentStream;
import org.apache.pdfbox.pdmodel.font.PDType1Font;
import org.apache.pdfbox.pdmodel.font.Standard14Fonts;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Offline tests: a generated 3-page PDF, a fake bag-of-words embedding model and a fake chat model, so no OpenAI
 * calls are made (free to run, no API key needed).
 */
class RagPipelineTest {

    /** Fake embeddings: one dimension per known word, value = how often it appears. Similar words -> similar vectors. */
    static final class WordEmbeddings implements EmbeddingModel {
        static final List<String> VOCAB = List.of("python", "loops", "lists", "sql", "joins", "tables",
                "internship", "months", "certificate");

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

    /** Fake LLM: records the prompt and answers with a fixed text. */
    static final class FakeLlm implements ChatModel {
        List<ChatMessage> lastPrompt;

        @Override
        public ChatResponse doChat(ChatRequest request) {
            lastPrompt = request.messages();
            return ChatResponse.builder().aiMessage(AiMessage.from("answer from context")).build();
        }
    }

    static Path makePdf(Path dir) throws IOException {
        String[][] pages = {
                {"PYTHON MODULE", "Python basics, loops and lists.", "More Python: loops, lists, functions."},
                {"SQL MODULE", "SQL joins and tables.", "Advanced SQL joins on large tables."},
                {"PROGRAM HIGHLIGHTS", "3 months internship with certificate.", "Internship certificate after 3 months."}};
        Path file = dir.resolve("agenda.pdf");
        try (PDDocument doc = new PDDocument()) {
            for (String[] lines : pages) {
                PDPage page = new PDPage();
                doc.addPage(page);
                try (PDPageContentStream cs = new PDPageContentStream(doc, page)) {
                    cs.beginText();
                    cs.setFont(new PDType1Font(Standard14Fonts.FontName.HELVETICA), 12);
                    cs.setLeading(16);
                    cs.newLineAtOffset(50, 700);
                    for (String line : lines) {
                        cs.showText(line);
                        cs.newLine();
                    }
                    cs.endText();
                }
            }
            doc.save(file.toFile());
        }
        return file;
    }

    @Test
    void loaderReturnsOneDocumentPerPageWithZeroBasedPageNumbers(@TempDir Path dir) throws IOException {
        List<Document> pages = PdfLoader.load(makePdf(dir));

        assertEquals(3, pages.size());
        assertTrue(pages.get(1).text().contains("SQL joins"));
        assertEquals(0, pages.get(0).metadata().getInteger("page"));
        assertEquals(2, pages.get(2).metadata().getInteger("page"));
        assertEquals(3, pages.get(0).metadata().getInteger("total_pages"));
        assertEquals("agenda.pdf", pages.get(0).metadata().getString("source"));
    }

    @Test
    void missingPdfGivesClearError() {
        IOException e = assertThrows(IOException.class, () -> PdfLoader.load(Path.of("no-such-file.pdf")));
        assertTrue(e.getMessage().startsWith("PDF not found"));
    }

    @Test
    void retrievalFindsTheRelevantPageAndPromptCarriesPageLabels(@TempDir Path dir) throws IOException {
        FakeLlm llm = new FakeLlm();
        RagPipeline rag = new RagPipeline(new WordEmbeddings(), llm);
        assertEquals(3, rag.index(PdfLoader.load(makePdf(dir))));
        assertEquals(WordEmbeddings.VOCAB.size() + 1, rag.dimension());

        // best match for a SQL question is the SQL page (page metadata 1 -> "Page 2")
        TextSegment best = rag.retrieve("Which SQL joins and tables?").getFirst().embedded();
        assertEquals(1, best.metadata().getInteger("page"));

        assertEquals("answer from context", rag.answer("What do they cover under python loops?"));
        String system = ((SystemMessage) llm.lastPrompt.getFirst()).text();
        assertTrue(system.startsWith("You are a helpful assistant. Answer the question using ONLY the context"));
        int pythonAt = system.indexOf("[Page 1]\nPYTHON MODULE");
        assertTrue(pythonAt > 0, system);
        assertTrue(pythonAt < system.indexOf("[Page 2]"), "most relevant chunk must come first");
        assertEquals("What do they cover under python loops?",
                ((dev.langchain4j.data.message.UserMessage) llm.lastPrompt.getLast()).singleText());
    }

    @Test
    void longPagesAreSplitIntoOverlappingChunksOfAtMost1000Chars() {
        String sentence = "Python lists and loops are covered in depth with practice. ";
        Document page = Document.from(sentence.repeat(60), new dev.langchain4j.data.document.Metadata().put("page", 4));
        RagPipeline rag = new RagPipeline(new WordEmbeddings(), new FakeLlm());

        int count = rag.index(List.of(page));

        assertTrue(count >= 4, "3600 chars should give several chunks, got " + count);
        for (TextSegment chunk : rag.chunks()) {
            assertTrue(chunk.text().length() <= RagPipeline.CHUNK_SIZE);
            assertEquals(4, chunk.metadata().getInteger("page"));
        }
        assertEquals("[Page 5]\n" + rag.chunks().getFirst().text(), RagPipeline.formatDocs(List.of(rag.chunks().getFirst())));
    }

    @Test
    void pdfWithoutTextIsRejected() {
        RagPipeline rag = new RagPipeline(new WordEmbeddings(), new FakeLlm());
        assertThrows(IllegalArgumentException.class, () -> rag.index(List.of(Document.from(" "))));
    }
}
