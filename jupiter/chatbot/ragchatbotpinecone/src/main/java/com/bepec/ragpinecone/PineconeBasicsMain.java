package com.bepec.ragpinecone;

import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.filter.Filter;

import java.util.List;

import static dev.langchain4j.store.embedding.filter.MetadataFilterBuilder.metadataKey;

/**
 * Pinecone basics - Java port of rag.ipynb. A small tour of a vector store, without PDFs or a chat model.
 *
 * <h2>Run</h2>
 * {@code mvn compile exec:java -Dexec.mainClass=com.bepec.ragpinecone.PineconeBasicsMain}
 * (needs OPENAI_API_KEY and PINECONE_API_KEY). It writes four small example vectors (doc1-doc4) into the default
 * namespace of the "langchain-demo" index - the same ids as the notebook, so running it again just overwrites them.
 *
 * <h2>What it shows</h2>
 * <ol>
 *   <li><b>Add documents with your own ids</b> - each text is embedded and stored with its metadata.</li>
 *   <li><b>Similarity search</b> - "What is Pinecone?" returns the stored texts closest in meaning.</li>
 *   <li><b>Search with scores and a metadata filter</b> - only vectors whose metadata matches (source = "notes")
 *       are considered; each result carries a similarity score (higher = closer, cosine).</li>
 *   <li><b>Retriever</b> - the same search wrapped as "give me the top k documents for this text", the building
 *       block of RAG.</li>
 *   <li><b>Add more documents</b> with different metadata (topic instead of source).</li>
 * </ol>
 */
public final class PineconeBasicsMain {

    private PineconeBasicsMain() {
    }

    /**
     * Entry point - runs the five steps of rag.ipynb in order and prints the results.
     * <ol>
     *   <li>Connect: OpenAI embedding model + Pinecone store for the default namespace (index created if missing,
     *       see {@link Config#pinecone}).</li>
     *   <li>{@link #addDocuments}: doc1 "Pinecone is a managed vector database." and doc2 "LangChain helps build LLM
     *       applications.", both with source = notes.</li>
     *   <li>{@link #similaritySearch} for "What is Pinecone?" (k = 2), printing text and metadata.</li>
     *   <li>{@link #similaritySearch} for "vector database" (k = 2) with filter source = notes, printing scores.</li>
     *   <li>Retriever: {@link #similaritySearch} for "Explain Pinecone" (k = 3).</li>
     *   <li>{@link #addDocuments}: doc3 (topic = agents) and doc4 (topic = rag).</li>
     * </ol>
     * Note: Pinecone is "eventually consistent" - vectors written a moment ago can take a second or two to show up in
     * searches, so the very first run may return fewer results than expected.
     *
     * @param args not used
     */
    public static void main(String[] args) {
        EmbeddingModel embeddings = Config.embeddings();
        EmbeddingStore<TextSegment> store = Config.pinecone(Config.DEFAULT_NAMESPACE);

        System.out.println("Added: " + addDocuments(embeddings, store,
                List.of("doc1", "doc2"),
                List.of(TextSegment.from("Pinecone is a managed vector database.", Metadata.from("source", "notes")),
                        TextSegment.from("LangChain helps build LLM applications.", Metadata.from("source", "notes")))));

        System.out.println("\nsimilarity_search(\"What is Pinecone?\", k=2):");
        for (var match : similaritySearch(embeddings, store, "What is Pinecone?", 2, null)) {
            System.out.println("  " + match.embedded().text() + " " + match.embedded().metadata().toMap());
        }

        System.out.println("\nsimilarity_search_with_score(\"vector database\", k=2, filter={source: notes}):");
        for (var match : similaritySearch(embeddings, store, "vector database", 2,
                metadataKey("source").isEqualTo("notes"))) {
            System.out.printf("  %.4f  %s%n", match.score(), match.embedded().text());
        }

        System.out.println("\nretriever.invoke(\"Explain Pinecone\"), k=3:");
        for (var match : similaritySearch(embeddings, store, "Explain Pinecone", 3, null)) {
            System.out.println("  [" + match.embeddingId() + "] " + match.embedded().text());
        }

        System.out.println("\nAdded: " + addDocuments(embeddings, store,
                List.of("doc3", "doc4"),
                List.of(TextSegment.from("Agentic AI systems plan and use tools.", Metadata.from("topic", "agents")),
                        TextSegment.from("RAG combines retrieval with generation.", Metadata.from("topic", "rag")))));
    }

    /**
     * {@code vector_store.add_documents(docs, ids=[...])}: embeds all texts in one call and upserts them with the
     * given ids. Using fixed ids makes the call repeatable - running it again overwrites instead of duplicating.
     *
     * @return the ids written, like the notebook's output {@code ['doc1', 'doc2']}
     */
    static List<String> addDocuments(EmbeddingModel embeddings, EmbeddingStore<TextSegment> store,
                                     List<String> ids, List<TextSegment> docs) {
        List<Embedding> vectors = embeddings.embedAll(docs).content();
        store.addAll(ids, vectors, docs);
        return ids;
    }

    /**
     * {@code similarity_search(query, k)} / {@code similarity_search_with_score(query, k, filter)} /
     * {@code retriever.invoke(query)} - they are all the same operation:
     * <ol>
     *   <li>Embed the query text.</li>
     *   <li>Ask the store for the {@code k} nearest vectors (cosine similarity), optionally only among vectors whose
     *       metadata matches {@code filter} - Pinecone applies the filter on the server, before ranking.</li>
     * </ol>
     *
     * @param filter metadata condition, e.g. {@code metadataKey("source").isEqualTo("notes")}, or null for none
     * @return matches, most similar first; each has the id, score and stored text + metadata
     */
    static List<EmbeddingMatch<TextSegment>> similaritySearch(EmbeddingModel embeddings,
                                                              EmbeddingStore<TextSegment> store,
                                                              String query, int k, Filter filter) {
        Embedding vector = embeddings.embedAll(List.of(TextSegment.from(query))).content().getFirst();
        return store.search(EmbeddingSearchRequest.builder()
                .queryEmbedding(vector)
                .maxResults(k)
                .filter(filter)
                .build()).matches();
    }
}
