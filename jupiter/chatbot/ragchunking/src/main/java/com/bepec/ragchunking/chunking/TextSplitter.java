package com.bepec.ragchunking.chunking;

import dev.langchain4j.data.segment.TextSegment;

import java.util.ArrayList;
import java.util.List;
import java.util.function.ToIntFunction;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Base class for the size-based splitters - a port of LangChain's {@code TextSplitter}.
 * <p>
 * Every size-based splitter works in two steps:
 * <ol>
 *   <li><b>split</b> the text into small pieces at a separator (a space, a blank line, ...), and</li>
 *   <li><b>merge</b> consecutive pieces back together until adding one more would exceed {@code chunkSize}
 *       ({@link #mergeSplits}). When a chunk is full, the start of the next chunk repeats up to
 *       {@code chunkOverlap} of the previous chunk's tail, so text cut at a boundary still appears whole in one
 *       of the chunks.</li>
 * </ol>
 * Sizes are measured with {@code lengthFunction}: characters by default (Python's {@code len}), or tokens.
 */
public abstract class TextSplitter {

    /** Where a matched separator goes: dropped, glued to the start of the next piece, or to the end of the previous. */
    public enum KeepSeparator { NONE, START, END }

    /** Python's {@code len}: counts characters (code points), not UTF-16 units. */
    public static final ToIntFunction<String> CHARACTERS = s -> s.codePointCount(0, s.length());

    protected final int chunkSize;
    protected final int chunkOverlap;
    protected final ToIntFunction<String> lengthFunction;
    protected final KeepSeparator keepSeparator;

    protected TextSplitter(int chunkSize, int chunkOverlap, ToIntFunction<String> lengthFunction,
                           KeepSeparator keepSeparator) {
        if (chunkSize <= 0) {
            throw new IllegalArgumentException("chunkSize must be > 0, got " + chunkSize);
        }
        if (chunkOverlap < 0 || chunkOverlap > chunkSize) {
            throw new IllegalArgumentException(
                    "chunkOverlap (" + chunkOverlap + ") must be between 0 and chunkSize (" + chunkSize + ")");
        }
        this.chunkSize = chunkSize;
        this.chunkOverlap = chunkOverlap;
        this.lengthFunction = lengthFunction;
        this.keepSeparator = keepSeparator;
    }

    /** {@code split_text(text)}. */
    public abstract List<String> splitText(String text);

    /** {@code split_documents(docs)}: splits each document; every chunk keeps a copy of its document's metadata. */
    public List<TextSegment> splitDocuments(List<TextSegment> documents) {
        List<TextSegment> out = new ArrayList<>();
        for (TextSegment doc : documents) {
            for (String chunk : splitText(doc.text())) {
                out.add(TextSegment.from(chunk, doc.metadata().copy()));
            }
        }
        return out;
    }

    /** {@code create_documents(texts)}. */
    public List<TextSegment> createDocuments(List<String> texts) {
        return splitDocuments(texts.stream().map(TextSegment::from).toList());
    }

    protected int length(String s) {
        return lengthFunction.applyAsInt(s);
    }

    /**
     * {@code _merge_splits}: greedily packs pieces into chunks of at most {@code chunkSize}, joining them with
     * {@code separator}; when a chunk is emitted, pieces are dropped from its front until what remains is no longer
     * than {@code chunkOverlap} - that remainder starts the next chunk (the overlap).
     */
    protected List<String> mergeSplits(List<String> splits, String separator) {
        int separatorLen = length(separator);
        List<String> docs = new ArrayList<>();
        List<String> current = new ArrayList<>();
        int total = 0;
        for (String piece : splits) {
            int len = length(piece);
            if (total + len + (current.isEmpty() ? 0 : separatorLen) > chunkSize) {
                if (!current.isEmpty()) {
                    addIfNotEmpty(docs, joinDocs(current, separator));
                    while (total > chunkOverlap
                            || (total + len + (current.isEmpty() ? 0 : separatorLen) > chunkSize && total > 0)) {
                        total -= length(current.getFirst()) + (current.size() > 1 ? separatorLen : 0);
                        current.removeFirst();
                    }
                }
            }
            current.add(piece);
            total += len + (current.size() > 1 ? separatorLen : 0);
        }
        addIfNotEmpty(docs, joinDocs(current, separator));
        return docs;
    }

    /** {@code _join_docs}: join and strip surrounding whitespace. */
    private static String joinDocs(List<String> docs, String separator) {
        return String.join(separator, docs).strip();
    }

    private static void addIfNotEmpty(List<String> docs, String doc) {
        if (!doc.isEmpty()) {
            docs.add(doc);
        }
    }

    /**
     * {@code _split_text_with_regex}: splits on a regex, optionally keeping each matched separator on the start
     * (START) or end (END) of a piece. An empty regex splits into single characters. Empty pieces are dropped.
     */
    static List<String> splitWithRegex(String text, String regex, KeepSeparator keep) {
        List<String> result = new ArrayList<>();
        if (regex.isEmpty()) {
            text.codePoints().forEach(cp -> result.add(new String(Character.toChars(cp))));
            return result;
        }
        List<String> pieces = new ArrayList<>();
        List<String> separators = new ArrayList<>();
        Matcher m = Pattern.compile(regex).matcher(text);
        int last = 0;
        while (m.find()) {
            pieces.add(text.substring(last, m.start()));
            separators.add(m.group());
            last = m.end();
        }
        pieces.add(text.substring(last));

        switch (keep) {
            case NONE -> result.addAll(pieces);
            case START -> {
                result.add(pieces.getFirst());
                for (int i = 0; i < separators.size(); i++) {
                    result.add(separators.get(i) + pieces.get(i + 1));
                }
            }
            case END -> {
                for (int i = 0; i < separators.size(); i++) {
                    result.add(pieces.get(i) + separators.get(i));
                }
                result.add(pieces.getLast());
            }
        }
        result.removeIf(String::isEmpty);
        return result;
    }
}
