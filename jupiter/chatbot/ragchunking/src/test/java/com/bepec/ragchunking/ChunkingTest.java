package com.bepec.ragchunking;

import com.bepec.ragchunking.chunking.CharacterTextSplitter;
import com.bepec.ragchunking.chunking.MarkdownHeaderTextSplitter;
import com.bepec.ragchunking.chunking.MarkdownHeaderTextSplitter.Header;
import com.bepec.ragchunking.chunking.RecursiveCharacterTextSplitter;
import com.bepec.ragchunking.chunking.RecursiveCharacterTextSplitter.Language;
import com.bepec.ragchunking.chunking.SemanticChunker;
import com.bepec.ragchunking.retrieval.ParentDocumentRetriever;
import com.knuddels.jtokkit.api.EncodingType;
import dev.langchain4j.data.segment.TextSegment;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static com.bepec.ragchunking.SampleData.SAMPLE_TEXT;
import static org.junit.jupiter.api.Assertions.*;

/** Each chunker must produce exactly what the notebook printed. */
class ChunkingTest {

    static final List<String> PARAGRAPHS = List.of(
            "Machine learning is a subset of AI. It enables systems to learn from data.",
            "Deep learning uses neural networks. These networks have many layers.",
            "RAG combines retrieval with generation. It reduces hallucinations in LLMs.");

    @Test
    void fixedSizeChunkingMatchesNotebook() {
        List<String> chunks = new CharacterTextSplitter(" ", 80, 20).splitText(SAMPLE_TEXT);

        assertEquals(List.of(
                "Machine learning is a subset of AI. It enables systems to learn from data.\n\nDeep",
                "from data.\n\nDeep learning uses neural networks. These networks have many",
                "networks have many layers.\n\nRAG combines retrieval with generation. It reduces",
                "It reduces hallucinations in LLMs."), chunks);
        chunks.forEach(c -> assertTrue(c.length() <= 80));
    }

    @Test
    void recursiveChunkingKeepsParagraphsTogether() {
        assertEquals(PARAGRAPHS, new RecursiveCharacterTextSplitter(80, 20).splitText(SAMPLE_TEXT));
    }

    @Test
    void recursiveChunkingFallsBackToWordsForLongParagraphs() {
        String text = "word ".repeat(50).strip();
        List<String> chunks = new RecursiveCharacterTextSplitter(30, 10).splitText(text);
        assertTrue(chunks.size() > 5);
        chunks.forEach(c -> assertTrue(c.length() <= 30, c));
        assertTrue(chunks.get(1).startsWith("word"));
    }

    @Test
    void tokenBasedChunkingMatchesNotebook() {
        var splitter = RecursiveCharacterTextSplitter.fromTiktokenEncoder(EncodingType.CL100K_BASE, 20, 5);
        assertEquals(PARAGRAPHS, splitter.splitText(SAMPLE_TEXT));
    }

    @Test
    void markdownHeaderChunkingMatchesNotebook() {
        var splitter = new MarkdownHeaderTextSplitter(List.of(new Header("#", "H1"), new Header("##", "H2")));
        var chunks = splitter.splitText(SampleData.MARKDOWN_TEXT);

        assertEquals(3, chunks.size());
        assertEquals("Machine learning learns from data.", chunks.get(0).content());
        assertEquals(Map.of("H1", "AI Basics"), chunks.get(0).metadata());
        assertEquals("Uses neural networks with many layers.", chunks.get(1).content());
        assertEquals("{'H1': 'AI Basics', 'H2': 'Deep Learning'}", Py.repr(chunks.get(1).metadata()));
        assertEquals("Combines retrieval and generation.", chunks.get(2).content());
        assertEquals("{'H1': 'AI Basics', 'H2': 'RAG'}", Py.repr(chunks.get(2).metadata()));
    }

    @Test
    void markdownHeadersInsideCodeBlocksAreIgnored() {
        var splitter = new MarkdownHeaderTextSplitter(List.of(new Header("#", "H1")));
        var chunks = splitter.splitText("# Title\ntext\n```\n# not a header\n```");
        assertEquals(1, chunks.size());
        assertTrue(chunks.getFirst().content().contains("# not a header"));
    }

    @Test
    void pythonCodeIsSplitBetweenFunctions() {
        List<String> chunks = RecursiveCharacterTextSplitter.fromLanguage(Language.PYTHON, 50, 0)
                .splitText(SampleData.PYTHON_CODE);
        assertEquals(List.of(
                "def greet(name):\n    return f\"Hello {name}\"",
                "def add(a, b):\n    return a + b"), chunks);
    }

    @Test
    void overlapLargerThanChunkSizeIsRejected() {
        assertThrows(IllegalArgumentException.class, () -> new RecursiveCharacterTextSplitter(100, 200));
    }

    @Test
    void semanticChunkerCutsAtTheBiggestTopicShift() {
        // ML sentences, then RAG sentences: the only big meaning-jump is between them
        String text = "Machine learning uses data. Deep learning uses neural networks. Neural networks have layers. "
                + "RAG combines retrieval with generation. RAG retrieval reduces hallucinations.";
        List<String> chunks = new SemanticChunker(new Fakes.WordEmbeddings(), 90).splitText(text);

        assertEquals(2, chunks.size(), chunks.toString());
        assertTrue(chunks.get(0).endsWith("Neural networks have layers."));
        assertTrue(chunks.get(1).startsWith("RAG combines"));
        assertEquals(text, String.join(" ", chunks));
    }

    @Test
    void parentDocumentRetrieverReturnsTheWholeParent() {
        var retriever = new ParentDocumentRetriever(new Fakes.WordEmbeddings(),
                new RecursiveCharacterTextSplitter(100, 20),
                new RecursiveCharacterTextSplitter(400, 50));
        retriever.addDocuments(List.of(TextSegment.from(SAMPLE_TEXT)));

        // the 211-character text fits in one parent, so the parent is the full text (as in the notebook)
        assertEquals(1, retriever.parentCount());
        assertEquals(SAMPLE_TEXT, retriever.retrieve("What is RAG?").getFirst().text());
    }

    @Test
    void parentDocumentRetrieverPicksTheMatchingParent() {
        var retriever = new ParentDocumentRetriever(new Fakes.WordEmbeddings(),
                new RecursiveCharacterTextSplitter(60, 10),
                new RecursiveCharacterTextSplitter(80, 0));
        retriever.addDocuments(List.of(TextSegment.from(SAMPLE_TEXT)));

        assertEquals(3, retriever.parentCount());
        List<TextSegment> parents = retriever.retrieve("rag retrieval generation hallucinations");
        assertEquals(PARAGRAPHS.get(2), parents.getFirst().text());
    }

    @Test
    void pythonReprMatchesPython() {
        assertEquals("'a\\nb'", Py.repr("a\nb"));
        assertEquals("\"it's\"", Py.repr("it's"));
    }
}
