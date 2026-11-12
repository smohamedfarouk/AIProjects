package com.bepec.ragchunking.retrieval;

import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.scoring.ScoringModel;

import java.util.Comparator;
import java.util.List;
import java.util.stream.IntStream;

/**
 * <b>Re-ranking RAG</b> - a port of {@code ContextualCompressionRetriever(base_compressor=CohereRerank(top_n=3),
 * base_retriever=...)}.
 * <p>
 * Vector search is fast but approximate: it compares a query vector and a document vector that were computed
 * separately. A re-ranker (cross-encoder) reads the query and each document <i>together</i> and scores how well
 * the document answers the query - much more accurate, but too slow to run over a whole collection. So: retrieve
 * generously (k=10), re-rank those few, keep the best {@code topN}. Each kept document gets its
 * {@code relevance_score} in the metadata.
 */
public class RerankingRetriever implements Retriever {

    private final Retriever baseRetriever;
    private final ScoringModel reranker;
    private final int topN;

    public RerankingRetriever(Retriever baseRetriever, ScoringModel reranker, int topN) {
        this.baseRetriever = baseRetriever;
        this.reranker = reranker;
        this.topN = topN;
    }

    @Override
    public List<TextSegment> retrieve(String query) {
        List<TextSegment> candidates = baseRetriever.retrieve(query);
        if (candidates.isEmpty()) {
            return List.of();
        }
        List<Double> scores = reranker.scoreAll(candidates, query).content();
        return IntStream.range(0, candidates.size()).boxed()
                .sorted(Comparator.comparingDouble((Integer i) -> scores.get(i)).reversed())
                .limit(topN)
                .map(i -> {
                    TextSegment doc = candidates.get(i);
                    return TextSegment.from(doc.text(), doc.metadata().copy().put("relevance_score", scores.get(i)));
                })
                .toList();
    }
}
