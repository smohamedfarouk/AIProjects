package com.bepec.ragchunking.retrieval;

import dev.langchain4j.data.segment.TextSegment;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * <b>Hybrid search</b> - a port of LangChain's {@code EnsembleRetriever}.
 * <p>
 * Runs several retrievers (typically BM25 keyword search + vector semantic search) and fuses their rankings with
 * <i>weighted Reciprocal Rank Fusion</i>: a document at rank r in a list gets {@code weight / (r + 60)}, and its
 * scores from all lists are added. Documents found high up by both retrievers win; scores are never compared
 * across retrievers, only ranks, so BM25 scores and cosine similarities don't need to be on the same scale.
 * <pre>
 * EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.4, 0.6])
 * </pre>
 */
public class EnsembleRetriever implements Retriever {

    static final int C = 60;

    private final List<Retriever> retrievers;
    private final List<Double> weights;

    public EnsembleRetriever(List<Retriever> retrievers, List<Double> weights) {
        if (retrievers.size() != weights.size()) {
            throw new IllegalArgumentException("Need one weight per retriever");
        }
        this.retrievers = List.copyOf(retrievers);
        this.weights = List.copyOf(weights);
    }

    @Override
    public List<TextSegment> retrieve(String query) {
        List<List<TextSegment>> rankings = new ArrayList<>();
        for (Retriever r : retrievers) {
            rankings.add(r.retrieve(query));
        }
        return fuse(rankings, weights);
    }

    /** Weighted RRF; documents are identified by their text. Ties keep first-seen order. */
    static List<TextSegment> fuse(List<List<TextSegment>> rankings, List<Double> weights) {
        Map<String, Double> score = new HashMap<>();
        Map<String, TextSegment> unique = new LinkedHashMap<>();
        for (int list = 0; list < rankings.size(); list++) {
            List<TextSegment> docs = rankings.get(list);
            for (int rank = 1; rank <= docs.size(); rank++) {
                TextSegment doc = docs.get(rank - 1);
                score.merge(doc.text(), weights.get(list) / (rank + C), Double::sum);
                unique.putIfAbsent(doc.text(), doc);
            }
        }
        return unique.values().stream()
                .sorted(Comparator.comparingDouble((TextSegment d) -> score.get(d.text())).reversed())
                .toList();
    }
}
