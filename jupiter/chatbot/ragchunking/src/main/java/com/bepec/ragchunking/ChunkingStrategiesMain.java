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
import dev.langchain4j.model.embedding.EmbeddingModel;

import java.util.List;

/**
 * Part 1 of the notebook, "Chunking Strategies": runs every chunking strategy on the same sample text and prints the
 * chunks the way the notebook does.
 *
 * <h2>Run</h2>
 * <pre>
 *   mvn compile exec:java
 * </pre>
 * Strategies 1-5 run offline. Semantic and parent-document chunking need OPENAI_API_KEY (environment or .env) and
 * are skipped without it.
 *
 * <h2>Notebook cell -> Java</h2>
 * <pre>
 * | Notebook                                              | Java                                          |
 * |-------------------------------------------------------|-----------------------------------------------|
 * | sample_text                                           | SampleData.SAMPLE_TEXT                        |
 * | CharacterTextSplitter(" ", 80, 20)                    | chunking.CharacterTextSplitter                |
 * | RecursiveCharacterTextSplitter(80, 20, separators)    | chunking.RecursiveCharacterTextSplitter       |
 * | RecursiveCharacterTextSplitter.from_tiktoken_encoder  | RecursiveCharacterTextSplitter.fromTiktoken.. |
 * | MarkdownHeaderTextSplitter([("#","H1"),("##","H2")])  | chunking.MarkdownHeaderTextSplitter           |
 * | RecursiveCharacterTextSplitter.from_language(PYTHON)  | RecursiveCharacterTextSplitter.fromLanguage   |
 * | SemanticChunker(OpenAIEmbeddings, percentile, 90)     | chunking.SemanticChunker                      |
 * | ParentDocumentRetriever(Chroma, InMemoryStore, ...)   | retrieval.ParentDocumentRetriever             |
 * </pre>
 */
public final class ChunkingStrategiesMain {

    private ChunkingStrategiesMain() {
    }

    public static void main(String[] args) {
        String text = SampleData.SAMPLE_TEXT;

        heading("1. Fixed-size (character) chunking");
        printChunks(new CharacterTextSplitter(" ", 80, 20).splitText(text), "");

        heading("2. Recursive character chunking");
        printChunks(new RecursiveCharacterTextSplitter(List.of("\n\n", "\n", " ", ""), 80, 20).splitText(text), "");

        heading("3. Token-based chunking (cl100k_base, 20 tokens, overlap 5)");
        printChunks(RecursiveCharacterTextSplitter.fromTiktokenEncoder(EncodingType.CL100K_BASE, 20, 5)
                .splitText(text), " (~tokens)");

        heading("4a. Document-specific chunking: Markdown headers");
        var markdown = new MarkdownHeaderTextSplitter(List.of(new Header("#", "H1"), new Header("##", "H2")));
        for (var chunk : markdown.splitText(SampleData.MARKDOWN_TEXT)) {
            System.out.println("Content: " + chunk.content());
            System.out.println("Metadata: " + Py.repr(chunk.metadata()) + "\n");
        }

        heading("4b. Document-specific chunking: Python code");
        List<String> code = RecursiveCharacterTextSplitter.fromLanguage(Language.PYTHON, 50, 0)
                .splitText(SampleData.PYTHON_CODE);
        for (int i = 0; i < code.size(); i++) {
            System.out.println("Chunk " + i + ":\n" + code.get(i) + "\n");
        }

        if (!Config.has("OPENAI_API_KEY")) {
            System.out.println("\nOPENAI_API_KEY not set - skipping semantic and parent-document chunking.");
            return;
        }
        EmbeddingModel embeddings = Config.embeddings();

        heading("5. Semantic chunking (percentile 90)");
        List<TextSegment> semantic = new SemanticChunker(embeddings, 90).createDocuments(List.of(text));
        for (int i = 0; i < semantic.size(); i++) {
            System.out.println("Chunk " + i + ": " + semantic.get(i).text() + "\n");
        }

        heading("6. Parent-document (small-to-big) chunking");
        // Parent = bigger context, child = smaller for precise matching; overlap must be smaller than chunk size
        var retriever = new ParentDocumentRetriever(embeddings,
                new RecursiveCharacterTextSplitter(100, 20),
                new RecursiveCharacterTextSplitter(400, 50));
        retriever.addDocuments(List.of(TextSegment.from(text)));
        System.out.println(retriever.retrieve("What is RAG?").getFirst().text());
    }

    private static void heading(String title) {
        System.out.println("\n=== " + title + " ===\n");
    }

    private static void printChunks(List<String> chunks, String label) {
        for (int i = 0; i < chunks.size(); i++) {
            System.out.println("Chunk " + i + label + ": " + Py.repr(chunks.get(i)) + "\n");
        }
    }
}
