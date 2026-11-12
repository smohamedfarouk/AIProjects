package com.bepec.ragchunking;

import dev.langchain4j.data.segment.TextSegment;

import java.util.List;

/** The example texts and documents used throughout the notebook. */
public final class SampleData {

    private SampleData() {
    }

    /** {@code sample_text}: three short paragraphs, two sentences each. */
    public static final String SAMPLE_TEXT = """
            Machine learning is a subset of AI. It enables systems to learn from data.

            Deep learning uses neural networks. These networks have many layers.

            RAG combines retrieval with generation. It reduces hallucinations in LLMs.""";

    /** {@code markdown_text} for document-specific (Markdown header) chunking. */
    public static final String MARKDOWN_TEXT = """
            # AI Basics
            Machine learning learns from data.

            ## Deep Learning
            Uses neural networks with many layers.

            ## RAG
            Combines retrieval and generation.""";

    /** {@code python_code} for code-aware chunking. */
    public static final String PYTHON_CODE = """
            def greet(name):
                return f"Hello {name}"

            def add(a, b):
                return a + b""";

    /** The three-document knowledge base used by the "Types of RAG" cells. */
    public static List<TextSegment> knowledgeBase() {
        return List.of(
                TextSegment.from("RAG combines retrieval with generation to reduce hallucinations."),
                TextSegment.from("LangGraph builds stateful, multi-step agent workflows."),
                TextSegment.from("Vector databases store embeddings for semantic search."));
    }
}
