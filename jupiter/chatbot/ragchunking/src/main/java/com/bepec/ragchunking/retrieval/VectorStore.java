package com.bepec.ragchunking.retrieval;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.List;

/**
 * In-memory vector store - plays the role of the notebook's {@code Chroma} (which is also in-memory there).
 * Each document is stored with its embedding; a search embeds the query and returns the documents whose vectors are
 * most similar (cosine similarity), i.e. a search by <i>meaning</i>, not by keyword.
 */
public class VectorStore {

    private final EmbeddingModel embeddingModel;
    private final InMemoryEmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();

    public VectorStore(EmbeddingModel embeddingModel) {
        this.embeddingModel = embeddingModel;
    }

    /** {@code Chroma.from_documents(docs, embeddings)}. */
    public static VectorStore fromDocuments(List<TextSegment> documents, EmbeddingModel embeddingModel) {
        VectorStore vs = new VectorStore(embeddingModel);
        vs.addDocuments(documents);
        return vs;
    }

    /** {@code add_documents(docs)}: embeds all documents in one batch call and stores them. */
    public void addDocuments(List<TextSegment> documents) {
        if (documents.isEmpty()) {
            return;
        }
        List<Embedding> vectors = embeddingModel.embedAll(documents).content();
        store.addAll(vectors, documents);
    }

    /** {@code similarity_search(query, k)}: the k most similar documents, best first. */
    public List<TextSegment> similaritySearch(String query, int k) {
        Embedding q = embeddingModel.embed(query).content();
        return store.search(EmbeddingSearchRequest.builder().queryEmbedding(q).maxResults(k).build())
                .matches().stream().map(EmbeddingMatch::embedded).toList();
    }

    /** {@code as_retriever(search_kwargs={"k": k})}. */
    public Retriever asRetriever(int k) {
        return query -> similaritySearch(query, k);
    }
}
