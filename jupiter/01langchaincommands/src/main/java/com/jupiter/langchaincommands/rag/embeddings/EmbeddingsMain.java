package com.jupiter.langchaincommands.rag.embeddings;

import com.jupiter.langchaincommands.common.Models;
import com.jupiter.langchaincommands.rag.RagPipeline;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.embedding.EmbeddingModel;

import java.util.Arrays;
import java.util.List;

/**
 * More LangChain commands, RAG lesson 3 - embeddings: turning text into numbers that capture meaning
 * (morelangchaincommands.ipynb cells 38-44). <b>Entry point.</b>
 * <p>
 * The RAG lessons: {@code rag.loader.WebLoaderMain} (load) -> {@code rag.chunking.ChunkingMain} (split) -> this class
 * (embed) -> {@code rag.vectorstore.VectorStoreSearchMain} (store and search) -> {@code rag.ragchain.RagChainMain}.
 *
 * <h2>Run</h2>
 * <pre>
 *   mvn compile exec:java@embeddings
 * </pre>
 * In IntelliJ, run this class's {@code main}. Needs OPENAI_API_KEY (environment or .env). Two small paid calls.
 *
 * <h2>What an embedding is</h2>
 * An embedding model turns a text into a fixed-length list of numbers (a vector) - 1536 numbers for
 * text-embedding-3-small. Texts with similar meaning get vectors that point in similar directions, so "how close are
 * these vectors" (cosine similarity) measures "how similar are these texts in meaning", even with different words.
 * That is what makes semantic search, and therefore RAG, possible.
 * <ul>
 *   <li>{@code embed_query(text)} - one text, typically the user's question.</li>
 *   <li>{@code embed_documents([...])} - many texts in one call, typically the chunks to be stored.</li>
 * </ul>
 * This class also prints how similar each document is to the question, to make the idea visible: the two
 * machine-learning sentences score higher than the pizza one.
 *
 * <h2>Notebook result</h2>
 * <pre>
 * len(query_vector) = 1536
 * First 5 values: [-0.0024356842041015625, -0.0127410888671875, -0.00664520263671875, -0.03155517578125, ...]
 * Embedded 3 documents
 * </pre>
 */
public final class EmbeddingsMain {

    static final String QUERY = "What is machine learning?";

    static final List<String> DOCS = List.of(
            "Machine learning is a subset of AI.",
            "Pizza is a popular Italian dish.",
            "Deep learning uses neural networks.");

    private EmbeddingsMain() {
    }

    /**
     * Entry point.
     * <ol>
     *   <li>Switch the console to UTF-8 and check OPENAI_API_KEY ({@link Models#init()}).</li>
     *   <li>{@code embed_query("What is machine learning?")}: print the vector length (1536) and its first 5
     *       values.</li>
     *   <li>{@code embed_documents(docs)} for the three sentences: print how many were embedded and each one's
     *       similarity to the question.</li>
     * </ol>
     *
     * @param args not used
     */
    public static void main(String[] args) {
        Models.init();
        EmbeddingModel embeddings = Models.openAiEmbeddings("text-embedding-3-small");

        Models.section("embeddings.embed_query(\"" + QUERY + "\")");
        Embedding query = RagPipeline.embedQuery(embeddings, QUERY);
        System.out.println("len(query_vector) = " + query.dimension());
        System.out.println("First 5 values: " + Arrays.toString(Arrays.copyOf(query.vector(), 5)));

        Models.section("embeddings.embed_documents(docs)");
        List<Embedding> docVectors = RagPipeline.embedDocuments(embeddings, DOCS);
        System.out.println("Embedded " + docVectors.size() + " documents\n");
        for (int i = 0; i < DOCS.size(); i++) {
            System.out.printf("similarity to the question %.3f  %s%n", RagPipeline.cosine(query, docVectors.get(i)), DOCS.get(i));
        }
    }
}
