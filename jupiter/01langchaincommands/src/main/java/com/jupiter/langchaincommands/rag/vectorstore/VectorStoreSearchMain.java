package com.jupiter.langchaincommands.rag.vectorstore;

import com.jupiter.langchaincommands.common.Models;
import com.jupiter.langchaincommands.rag.RagPipeline;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.List;

/**
 * More LangChain commands, RAG lesson 4 - a vector store: store embedded documents and find the ones closest in
 * meaning to a question (morelangchaincommands.ipynb cells 45-51). <b>Entry point.</b>
 * <p>
 * The RAG lessons: {@code rag.loader.WebLoaderMain} (load) -> {@code rag.chunking.ChunkingMain} (split) ->
 * {@code rag.embeddings.EmbeddingsMain} (embed) -> this class (store and search) -> {@code rag.ragchain.RagChainMain}.
 *
 * <h2>Run</h2>
 * <pre>
 *   mvn compile exec:java@vector-search
 * </pre>
 * In IntelliJ, run this class's {@code main}. Needs OPENAI_API_KEY (environment or .env). Two small paid calls.
 *
 * <h2>How it works</h2>
 * <pre>
 *   FAISS.from_documents(docs, embeddings)   -> embed every document, keep (vector, text) pairs in memory
 *   similarity_search(query, k=2)            -> embed the query, return the 2 documents with the closest vectors
 * </pre>
 * The notebook uses FAISS (a fast in-memory vector index from Meta); LangChain4j's {@link InMemoryEmbeddingStore} plays
 * the same role here. Both live in memory and are rebuilt on every run; a database like Pinecone keeps them (see the
 * ragchatbotpinecone project).
 * <p>
 * Note the result: "where can i learn data science" finds "BEPEC offers Data Science courses in Bangalore." first,
 * although "learn" and "where" appear in neither text - the search is by meaning, not by keywords.
 *
 * <h2>Notebook result</h2>
 * <pre>
 * [Document(page_content='BEPEC offers Data Science courses in Bangalore.'),
 *  Document(page_content='RAG combines retrieval with text generation in Data Science')]
 * </pre>
 */
public final class VectorStoreSearchMain {

    static final List<String> DOCS = List.of(
            "BEPEC offers Data Science courses in Bangalore.",
            "LangChain helps build LLM applications quickly.",
            "Python is the most popular language for AI.",
            "RAG combines retrieval with text generation in Data Science");

    static final String QUERY = "where can i learn data science";

    private VectorStoreSearchMain() {
    }

    /**
     * Entry point.
     * <ol>
     *   <li>Switch the console to UTF-8 and check OPENAI_API_KEY ({@link Models#init()}).</li>
     *   <li>Build the store from the four documents with text-embedding-3-small ({@link RagPipeline#fromDocuments}).</li>
     *   <li>Search "where can i learn data science" with k = 2 and print each result with its score.</li>
     * </ol>
     *
     * @param args not used
     */
    public static void main(String[] args) {
        Models.init();
        EmbeddingModel embeddings = Models.openAiEmbeddings("text-embedding-3-small");

        Models.section("vectorstore.similarity_search(\"" + QUERY + "\", k=2)");
        InMemoryEmbeddingStore<TextSegment> store = RagPipeline.fromDocuments(DOCS, embeddings);
        for (EmbeddingMatch<TextSegment> match : RagPipeline.similaritySearch(store, embeddings, QUERY, 2)) {
            System.out.printf("%.3f  %s%n", match.score(), match.embedded().text());
        }
    }
}
