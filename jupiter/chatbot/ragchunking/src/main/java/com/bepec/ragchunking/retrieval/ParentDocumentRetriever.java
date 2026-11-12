package com.bepec.ragchunking.retrieval;

import com.bepec.ragchunking.chunking.TextSplitter;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;

/**
 * <b>Parent-document (small-to-big) chunking</b> - a port of LangChain's {@code ParentDocumentRetriever}.
 * <p>
 * Solves a core tension: small chunks match queries precisely, but large chunks give the LLM better context. So it
 * stores small "child" chunks for searching, and when one matches, it returns the larger "parent" chunk it came
 * from:
 * <pre>
 *  add:    document --parent splitter(400)--> parents --(each gets an id, kept in the docstore)
 *                                                  --child splitter(100)--> children tagged with doc_id
 *                                                  --> embedded into the vector store
 *  search: query --> best child chunks --> their doc_ids (unique, in order) --> parent chunks from the docstore
 * </pre>
 * The overlap of each splitter must be smaller than its chunk size.
 */
public class ParentDocumentRetriever implements Retriever {

    static final String ID_KEY = "doc_id";
    /** Chroma's default k for similarity_search. */
    static final int CHILD_K = 4;

    private final VectorStore vectorStore;
    private final Map<String, TextSegment> docstore = new LinkedHashMap<>(); // InMemoryStore
    private final TextSplitter childSplitter;
    private final TextSplitter parentSplitter;

    public ParentDocumentRetriever(EmbeddingModel embeddingModel, TextSplitter childSplitter,
                                   TextSplitter parentSplitter) {
        this.vectorStore = new VectorStore(embeddingModel);
        this.childSplitter = childSplitter;
        this.parentSplitter = parentSplitter;
    }

    /** {@code retriever.add_documents(docs)}. */
    public void addDocuments(List<TextSegment> documents) {
        List<TextSegment> parents = parentSplitter != null ? parentSplitter.splitDocuments(documents) : documents;
        List<TextSegment> children = new ArrayList<>();
        for (TextSegment parent : parents) {
            String id = UUID.randomUUID().toString();
            docstore.put(id, parent);
            for (TextSegment child : childSplitter.splitDocuments(List.of(parent))) {
                child.metadata().put(ID_KEY, id);
                children.add(child);
            }
        }
        vectorStore.addDocuments(children);
    }

    /** {@code retriever.invoke(query)}: search the children, return their parents. */
    @Override
    public List<TextSegment> retrieve(String query) {
        Set<String> ids = new LinkedHashSet<>();
        for (TextSegment child : vectorStore.similaritySearch(query, CHILD_K)) {
            ids.add(child.metadata().getString(ID_KEY));
        }
        return ids.stream().map(docstore::get).filter(java.util.Objects::nonNull).toList();
    }

    /** Number of parent chunks stored. */
    public int parentCount() {
        return docstore.size();
    }
}
