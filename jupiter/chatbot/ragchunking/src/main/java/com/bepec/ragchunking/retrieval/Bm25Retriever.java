package com.bepec.ragchunking.retrieval;

import dev.langchain4j.data.segment.TextSegment;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.IntStream;

/**
 * <b>Keyword search</b> - a port of LangChain's {@code BM25Retriever} (which uses {@code rank_bm25.BM25Okapi}).
 * <p>
 * BM25 is classic search-engine scoring: a document scores higher the more often it contains the query's words
 * (with diminishing returns, {@code k1}), the rarer those words are across all documents (IDF), and the shorter the
 * document is ({@code b}). It finds exact terms - names, codes, acronyms - that embeddings can miss. Like the
 * LangChain default, text is tokenised by splitting on whitespace (case-sensitive).
 */
public class Bm25Retriever implements Retriever {

    static final double K1 = 1.5;
    static final double B = 0.75;
    static final double EPSILON = 0.25;

    private final List<TextSegment> documents;
    private final Function<String, List<String>> preprocess;
    private final List<Map<String, Integer>> termFrequencies = new ArrayList<>();
    private final int[] docLengths;
    private final double avgDocLength;
    private final Map<String, Double> idf = new HashMap<>();
    private int k = 4;

    public Bm25Retriever(List<TextSegment> documents, Function<String, List<String>> preprocess) {
        this.documents = List.copyOf(documents);
        this.preprocess = preprocess;
        this.docLengths = new int[documents.size()];

        Map<String, Integer> docsContaining = new HashMap<>();
        long totalLength = 0;
        for (int i = 0; i < documents.size(); i++) {
            List<String> tokens = preprocess.apply(documents.get(i).text());
            docLengths[i] = tokens.size();
            totalLength += tokens.size();
            Map<String, Integer> freq = new HashMap<>();
            tokens.forEach(t -> freq.merge(t, 1, Integer::sum));
            termFrequencies.add(freq);
            freq.keySet().forEach(t -> docsContaining.merge(t, 1, Integer::sum));
        }
        this.avgDocLength = documents.isEmpty() ? 0 : (double) totalLength / documents.size();

        // idf = log(N - n + 0.5) - log(n + 0.5); negative idfs (words in most docs) are floored at eps * average idf
        int n = documents.size();
        double idfSum = 0;
        List<String> negative = new ArrayList<>();
        for (var e : docsContaining.entrySet()) {
            double value = Math.log(n - e.getValue() + 0.5) - Math.log(e.getValue() + 0.5);
            idf.put(e.getKey(), value);
            idfSum += value;
            if (value < 0) {
                negative.add(e.getKey());
            }
        }
        double eps = idf.isEmpty() ? 0 : EPSILON * idfSum / idf.size();
        negative.forEach(word -> idf.put(word, eps));
    }

    /** {@code BM25Retriever.from_documents(docs)} with the default whitespace tokenizer. */
    public static Bm25Retriever fromDocuments(List<TextSegment> documents) {
        return new Bm25Retriever(documents, text -> List.of(text.strip().split("\\s+")));
    }

    /** {@code bm25_retriever.k = k}. */
    public Bm25Retriever setK(int k) {
        this.k = k;
        return this;
    }

    /** BM25 score of every document for the query. */
    public double[] scores(String query) {
        double[] scores = new double[documents.size()];
        for (String q : preprocess.apply(query)) {
            double weight = idf.getOrDefault(q, 0.0);
            for (int i = 0; i < documents.size(); i++) {
                int tf = termFrequencies.get(i).getOrDefault(q, 0);
                scores[i] += weight * (tf * (K1 + 1)
                        / (tf + K1 * (1 - B + B * docLengths[i] / avgDocLength)));
            }
        }
        return scores;
    }

    /** The k best documents ({@code np.argsort(scores)[::-1][:k]}: on a tie the later document comes first). */
    @Override
    public List<TextSegment> retrieve(String query) {
        double[] scores = scores(query);
        return IntStream.range(0, documents.size()).boxed()
                .sorted(Comparator.<Integer>comparingDouble(i -> scores[i]).thenComparing(i -> i).reversed())
                .limit(k)
                .map(documents::get)
                .toList();
    }
}
