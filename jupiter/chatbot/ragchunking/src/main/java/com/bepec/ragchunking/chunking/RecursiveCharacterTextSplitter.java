package com.bepec.ragchunking.chunking;

import com.knuddels.jtokkit.Encodings;
import com.knuddels.jtokkit.api.Encoding;
import com.knuddels.jtokkit.api.EncodingType;

import java.util.ArrayList;
import java.util.List;
import java.util.function.ToIntFunction;
import java.util.regex.Pattern;

/**
 * <b>Recursive character chunking</b> - a port of LangChain's {@code RecursiveCharacterTextSplitter}.
 * <p>
 * Tries to split on the largest natural boundary first (paragraphs, {@code "\n\n"}), and only if a piece is still
 * too big does it fall back to smaller separators (lines {@code "\n"} -> spaces {@code " "} -> characters
 * {@code ""}). This keeps related text together far better than fixed-size chunking: with chunk_size=80 every
 * paragraph of the sample text becomes exactly one chunk.
 * <pre>
 * RecursiveCharacterTextSplitter(chunk_size=80, chunk_overlap=20, separators=["\n\n", "\n", " ", ""])
 * </pre>
 * Two factory methods cover the notebook's other variants:
 * <ul>
 *   <li>{@link #fromTiktokenEncoder} - <b>token-based chunking</b>: the same algorithm, but sizes are counted in
 *       tokens (what the LLM actually reads) with the {@code cl100k_base} tokenizer of GPT-3.5/4, giving precise
 *       control over the context budget.</li>
 *   <li>{@link #fromLanguage} - <b>code-aware chunking</b>: language-specific separators such as
 *       {@code "\ndef "} and {@code "\nclass "}, so code is cut between functions instead of inside them.</li>
 * </ul>
 */
public class RecursiveCharacterTextSplitter extends TextSplitter {

    /** Default separators: paragraph, line, word, character. */
    public static final List<String> DEFAULT_SEPARATORS = List.of("\n\n", "\n", " ", "");

    /** {@code langchain.text_splitter.Language} - the separators (regexes) used for each language. */
    public enum Language {
        PYTHON(List.of("\nclass ", "\ndef ", "\n\tdef ", "\n\n", "\n", " ", "")),
        JAVA(List.of("\nclass ", "\npublic ", "\nprotected ", "\nprivate ", "\nstatic ", "\nif ", "\nfor ",
                "\nwhile ", "\nswitch ", "\ncase ", "\n\n", "\n", " ", "")),
        MARKDOWN(List.of("\n#{1,6} ", "```\n", "\n\\*\\*\\*+\n", "\n---+\n", "\n___+\n", "\n\n", "\n", " ", ""));

        final List<String> separators;

        Language(List<String> separators) {
            this.separators = separators;
        }

        public List<String> separators() {
            return separators;
        }
    }

    private final List<String> separators;
    private final boolean separatorsAreRegex;

    /** {@code RecursiveCharacterTextSplitter(chunk_size, chunk_overlap)} with the default separators. */
    public RecursiveCharacterTextSplitter(int chunkSize, int chunkOverlap) {
        this(DEFAULT_SEPARATORS, chunkSize, chunkOverlap);
    }

    public RecursiveCharacterTextSplitter(List<String> separators, int chunkSize, int chunkOverlap) {
        this(separators, false, chunkSize, chunkOverlap, CHARACTERS);
    }

    public RecursiveCharacterTextSplitter(List<String> separators, boolean separatorsAreRegex, int chunkSize,
                                          int chunkOverlap, ToIntFunction<String> lengthFunction) {
        super(chunkSize, chunkOverlap, lengthFunction, KeepSeparator.START);
        this.separators = List.copyOf(separators);
        this.separatorsAreRegex = separatorsAreRegex;
    }

    /**
     * {@code RecursiveCharacterTextSplitter.from_tiktoken_encoder(encoding_name=..., chunk_size, chunk_overlap)}:
     * chunk_size and chunk_overlap are token counts.
     */
    public static RecursiveCharacterTextSplitter fromTiktokenEncoder(EncodingType encodingType, int chunkSize,
                                                                     int chunkOverlap) {
        Encoding encoding = Encodings.newLazyEncodingRegistry().getEncoding(encodingType);
        return new RecursiveCharacterTextSplitter(DEFAULT_SEPARATORS, false, chunkSize, chunkOverlap,
                encoding::countTokens);
    }

    /** {@code RecursiveCharacterTextSplitter.from_language(language, chunk_size, chunk_overlap)}. */
    public static RecursiveCharacterTextSplitter fromLanguage(Language language, int chunkSize, int chunkOverlap) {
        return new RecursiveCharacterTextSplitter(language.separators, true, chunkSize, chunkOverlap, CHARACTERS);
    }

    @Override
    public List<String> splitText(String text) {
        return splitText(text, separators);
    }

    /**
     * {@code _split_text}: pick the first separator that occurs in the text, split on it, keep pieces that fit,
     * and recurse with the remaining (finer) separators into any piece that is still too big.
     */
    private List<String> splitText(String text, List<String> separators) {
        List<String> finalChunks = new ArrayList<>();
        String separator = separators.getLast();
        List<String> newSeparators = List.of();
        for (int i = 0; i < separators.size(); i++) {
            String s = separators.get(i);
            if (s.isEmpty()) {
                separator = s;
                break;
            }
            if (Pattern.compile(regex(s)).matcher(text).find()) {
                separator = s;
                newSeparators = separators.subList(i + 1, separators.size());
                break;
            }
        }

        List<String> splits = splitWithRegex(text, regex(separator), keepSeparator);
        String mergeSeparator = keepSeparator != KeepSeparator.NONE ? "" : separator;
        List<String> goodSplits = new ArrayList<>();
        for (String s : splits) {
            if (length(s) < chunkSize) {
                goodSplits.add(s);
            } else {
                if (!goodSplits.isEmpty()) {
                    finalChunks.addAll(mergeSplits(goodSplits, mergeSeparator));
                    goodSplits = new ArrayList<>();
                }
                if (newSeparators.isEmpty()) {
                    finalChunks.add(s);
                } else {
                    finalChunks.addAll(splitText(s, newSeparators));
                }
            }
        }
        if (!goodSplits.isEmpty()) {
            finalChunks.addAll(mergeSplits(goodSplits, mergeSeparator));
        }
        return finalChunks;
    }

    private String regex(String separator) {
        return separatorsAreRegex || separator.isEmpty() ? separator : Pattern.quote(separator);
    }
}
