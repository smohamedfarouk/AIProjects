package com.bepec.ragchunking.chunking;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Deque;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * <b>Document-specific chunking</b> - a port of LangChain's {@code MarkdownHeaderTextSplitter}.
 * <p>
 * Splits according to the document's structure instead of blind size rules: every section under a Markdown header
 * becomes one chunk, and the headers above it are kept as metadata - e.g. the text under {@code ## RAG} gets
 * {@code {'H1': 'AI Basics', 'H2': 'RAG'}}. That metadata can later be used to filter searches or to show the
 * reader where an answer came from.
 * <pre>
 * MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "H1"), ("##", "H2")])
 * </pre>
 * Like the LangChain default, header lines are removed from the content ({@code strip_headers=True}), lines with the
 * same metadata are combined into one chunk, and lines inside {@code ```} / {@code ~~~} code blocks are never read
 * as headers.
 */
public class MarkdownHeaderTextSplitter {

    /** One section: its text and the headers it sits under, outermost first. */
    public record Chunk(String content, Map<String, String> metadata) {
    }

    /** A header marker ({@code "#"}, {@code "##"}, ...) and the metadata key to store its title under. */
    public record Header(String marker, String name) {
    }

    private record Line(String content, Map<String, String> metadata) {
    }

    private record OpenHeader(int level, String name) {
    }

    private final List<Header> headersToSplitOn;

    public MarkdownHeaderTextSplitter(List<Header> headersToSplitOn) {
        // longest marker first, so "##" is not mistaken for "#"
        this.headersToSplitOn = headersToSplitOn.stream()
                .sorted(Comparator.comparingInt((Header h) -> h.marker().length()).reversed())
                .toList();
    }

    public List<Chunk> splitText(String text) {
        List<Line> lines = new ArrayList<>();
        List<String> currentContent = new ArrayList<>();
        Map<String, String> currentMetadata = new LinkedHashMap<>();
        Map<String, String> headerMetadata = new LinkedHashMap<>();
        Deque<OpenHeader> headerStack = new ArrayDeque<>();
        boolean inCodeBlock = false;
        String openingFence = "";

        for (String raw : text.split("\n", -1)) {
            String line = raw.strip().replaceAll("\\p{C}", "");

            if (!inCodeBlock) {
                if (line.startsWith("```") && countOf(line, "```") == 1) {
                    inCodeBlock = true;
                    openingFence = "```";
                } else if (line.startsWith("~~~")) {
                    inCodeBlock = true;
                    openingFence = "~~~";
                }
            } else if (line.startsWith(openingFence)) {
                inCodeBlock = false;
                openingFence = "";
            }
            if (inCodeBlock) {
                currentContent.add(line);
                continue;
            }

            Header matched = null;
            for (Header h : headersToSplitOn) {
                if (line.startsWith(h.marker())
                        && (line.length() == h.marker().length() || line.charAt(h.marker().length()) == ' ')) {
                    matched = h;
                    break;
                }
            }

            if (matched != null) {
                int level = countOf(matched.marker(), "#");
                // a header closes every open header at the same or a deeper level
                while (!headerStack.isEmpty() && headerStack.peek().level() >= level) {
                    headerMetadata.remove(headerStack.pop().name());
                }
                headerStack.push(new OpenHeader(level, matched.name()));
                headerMetadata.put(matched.name(), line.substring(matched.marker().length()).strip());

                if (!currentContent.isEmpty()) {
                    lines.add(new Line(String.join("\n", currentContent), new LinkedHashMap<>(currentMetadata)));
                    currentContent.clear();
                }
            } else if (!line.isEmpty()) {
                currentContent.add(line);
            } else if (!currentContent.isEmpty()) {
                lines.add(new Line(String.join("\n", currentContent), new LinkedHashMap<>(currentMetadata)));
                currentContent.clear();
            }
            currentMetadata = new LinkedHashMap<>(headerMetadata);
        }
        if (!currentContent.isEmpty()) {
            lines.add(new Line(String.join("\n", currentContent), currentMetadata));
        }
        return aggregate(lines);
    }

    /** {@code aggregate_lines_to_chunks}: consecutive blocks with the same headers become one chunk. */
    private static List<Chunk> aggregate(List<Line> lines) {
        List<Chunk> chunks = new ArrayList<>();
        for (Line line : lines) {
            if (!chunks.isEmpty() && chunks.getLast().metadata().equals(line.metadata())) {
                Chunk last = chunks.removeLast();
                chunks.add(new Chunk(last.content() + "  \n" + line.content(), last.metadata()));
            } else {
                chunks.add(new Chunk(line.content(), line.metadata()));
            }
        }
        return chunks.stream()
                .map(c -> new Chunk(c.content(), java.util.Collections.unmodifiableMap(c.metadata())))
                .toList();
    }

    private static int countOf(String text, String part) {
        int count = 0;
        for (int i = text.indexOf(part); i >= 0; i = text.indexOf(part, i + part.length())) {
            count++;
        }
        return count;
    }
}
