package com.bepec.ragchunking.chunking;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.CosineSimilarity;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * <b>Semantic chunking</b> - a port of {@code langchain_experimental}'s {@code SemanticChunker} with the
 * {@code "percentile"} breakpoint type.
 * <p>
 * Instead of any fixed size, it embeds each sentence and measures how similar consecutive sentences are. When the
 * topic shifts (similarity drops sharply), it cuts there, so chunks break at meaning boundaries:
 * <ol>
 *   <li>Split the text into sentences (after {@code . ? !} followed by whitespace).</li>
 *   <li>Give each sentence a little context - it is embedded together with its neighbour on each side
 *       ({@code buffer_size=1}), which smooths out very short sentences.</li>
 *   <li>Compute the cosine <i>distance</i> (1 - similarity) between each sentence and the next.</li>
 *   <li>Cut wherever the distance is above the {@code breakpointPercentile}-th percentile of all distances -
 *       with 90, only the biggest ~10% of meaning-jumps become chunk boundaries.</li>
 * </ol>
 * <pre>
 * SemanticChunker(OpenAIEmbeddings(model="text-embedding-3-small"),
 *                 breakpoint_threshold_type="percentile", breakpoint_threshold_amount=90)
 * </pre>
 */
public class SemanticChunker {

    static final String SENTENCE_SPLIT_REGEX = "(?<=[.?!])\\s+";

    private final EmbeddingModel embeddingModel;
    private final double breakpointPercentile;
    private final int bufferSize;

    public SemanticChunker(EmbeddingModel embeddingModel, double breakpointPercentile) {
        this(embeddingModel, breakpointPercentile, 1);
    }

    public SemanticChunker(EmbeddingModel embeddingModel, double breakpointPercentile, int bufferSize) {
        this.embeddingModel = embeddingModel;
        this.breakpointPercentile = breakpointPercentile;
        this.bufferSize = bufferSize;
    }

    /** {@code create_documents(texts)}. */
    public List<TextSegment> createDocuments(List<String> texts) {
        List<TextSegment> out = new ArrayList<>();
        for (String text : texts) {
            splitText(text).forEach(chunk -> out.add(TextSegment.from(chunk)));
        }
        return out;
    }

    public List<String> splitText(String text) {
        List<String> sentences = List.of(text.split(SENTENCE_SPLIT_REGEX));
        if (sentences.size() <= 1) {
            return sentences;
        }
        double[] distances = distancesBetweenSentences(sentences);
        double threshold = percentile(distances, breakpointPercentile);

        List<String> chunks = new ArrayList<>();
        int start = 0;
        for (int i = 0; i < distances.length; i++) {
            if (distances[i] > threshold) {
                chunks.add(String.join(" ", sentences.subList(start, i + 1)));
                start = i + 1;
            }
        }
        if (start < sentences.size()) {
            chunks.add(String.join(" ", sentences.subList(start, sentences.size())));
        }
        return chunks;
    }

    /** distance[i] = 1 - cosine(embedding of sentence i with its neighbours, same for sentence i + 1). */
    double[] distancesBetweenSentences(List<String> sentences) {
        List<TextSegment> combined = new ArrayList<>();
        for (int i = 0; i < sentences.size(); i++) {
            int from = Math.max(0, i - bufferSize);
            int to = Math.min(sentences.size(), i + bufferSize + 1);
            combined.add(TextSegment.from(String.join(" ", sentences.subList(from, to))));
        }
        List<Embedding> embeddings = embeddingModel.embedAll(combined).content();
        double[] distances = new double[sentences.size() - 1];
        for (int i = 0; i < distances.length; i++) {
            distances[i] = 1 - CosineSimilarity.between(embeddings.get(i), embeddings.get(i + 1));
        }
        return distances;
    }

    /** {@code numpy.percentile} with its default linear interpolation. */
    static double percentile(double[] values, double percent) {
        double[] sorted = values.clone();
        Arrays.sort(sorted);
        double rank = percent / 100.0 * (sorted.length - 1);
        int lower = (int) Math.floor(rank);
        int upper = (int) Math.ceil(rank);
        return sorted[lower] + (sorted[upper] - sorted[lower]) * (rank - lower);
    }
}
