package com.bepec.ragchunking.chunking;

import java.util.List;
import java.util.regex.Pattern;

/**
 * <b>Fixed-size (character) chunking</b> - a port of LangChain's {@code CharacterTextSplitter}.
 * <p>
 * Cuts text into blocks of a fixed character count. It splits at ONE separator only (here a space) and packs the
 * pieces into chunks of at most {@code chunkSize}; it doesn't care about meaning or sentence boundaries, so a chunk
 * can end in the middle of a sentence ({@code '... learn from data.\n\nDeep'}).
 * <pre>
 * CharacterTextSplitter(separator=" ", chunk_size=80, chunk_overlap=20, length_function=len)
 * </pre>
 */
public class CharacterTextSplitter extends TextSplitter {

    private final String separator;
    private final boolean separatorIsRegex;

    public CharacterTextSplitter(String separator, int chunkSize, int chunkOverlap) {
        this(separator, false, chunkSize, chunkOverlap, CHARACTERS, KeepSeparator.NONE);
    }

    public CharacterTextSplitter(String separator, boolean separatorIsRegex, int chunkSize, int chunkOverlap,
                                 java.util.function.ToIntFunction<String> lengthFunction, KeepSeparator keepSeparator) {
        super(chunkSize, chunkOverlap, lengthFunction, keepSeparator);
        this.separator = separator;
        this.separatorIsRegex = separatorIsRegex;
    }

    @Override
    public List<String> splitText(String text) {
        String pattern = separatorIsRegex ? separator : (separator.isEmpty() ? "" : Pattern.quote(separator));
        List<String> splits = splitWithRegex(text, pattern, keepSeparator);
        boolean lookaround = separatorIsRegex && (separator.startsWith("(?=") || separator.startsWith("(?<!")
                || separator.startsWith("(?<=") || separator.startsWith("(?!"));
        String mergeSeparator = keepSeparator != KeepSeparator.NONE || lookaround ? "" : separator;
        return mergeSplits(splits, mergeSeparator);
    }
}
