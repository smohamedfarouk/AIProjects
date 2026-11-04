package com.jupiter.langchaincommands.rag;

import com.jupiter.langchaincommands.chat.chatprompttemplate.PromptChain;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import javax.swing.text.MutableAttributeSet;
import javax.swing.text.html.HTML;
import javax.swing.text.html.HTMLEditorKit;
import javax.swing.text.html.parser.ParserDelegator;
import java.io.IOException;
import java.io.StringReader;
import java.io.UncheckedIOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * All RAG building blocks in one class, in pipeline order - the methods behind the five RAG lessons
 * ({@code rag.loader.WebLoaderMain}, {@code rag.chunking.ChunkingMain}, {@code rag.embeddings.EmbeddingsMain},
 * {@code rag.vectorstore.VectorStoreSearchMain} and {@code rag.ragchain.RagChainMain}), which each demonstrate one
 * step by calling the methods here.
 *
 * <h2>The pipeline</h2>
 * <pre>
 *  1. LOAD      loadWebPage(url) / fromHtml(html, url)      web page -> Document (text + metadata)
 *  2. CHUNK     split(text, size, overlap) / splitDocument  Document -> small, overlapping chunks
 *  3. EMBED     embedQuery / embedDocuments / cosine        text -> vector of numbers that captures meaning
 *  4. STORE     fromDocuments / fromSegments                 vectors + texts -> in-memory vector store
 *     SEARCH    similaritySearch(store, ..., query, k)       question -> the k chunks closest in meaning
 *  5. ANSWER    formatDocs + answer(...)                     retrieved chunks -> context -> model -> answer
 * </pre>
 * Example - answer a question about a web page with all five steps:
 * <pre>
 *   Document page = RagPipeline.loadWebPage("https://python.langchain.com/docs/introduction/");
 *   List&lt;TextSegment&gt; chunks = RagPipeline.splitDocument(page, 500, 50);
 *   var store = RagPipeline.fromSegments(chunks, embeddings);
 *   String answer = RagPipeline.answer(store, embeddings, llm, "What is LangChain?", 3);
 * </pre>
 *
 * <h2>Python -> Java</h2>
 * <pre>
 * | Step | LangChain (Python)                                   | Method here                          |
 * |------|------------------------------------------------------|--------------------------------------|
 * | 1    | WebBaseLoader(url).load()                            | loadWebPage(url)                     |
 * | 2    | RecursiveCharacterTextSplitter(...).split_text(text) | split(text, chunkSize, overlap)      |
 * | 3    | OpenAIEmbeddings.embed_query / embed_documents       | embedQuery / embedDocuments          |
 * | 4    | FAISS.from_documents(docs, embeddings)               | fromDocuments(docs, embeddings)      |
 * | 4    | vectorstore.similarity_search(query, k)              | similaritySearch(store, ..., k)      |
 * | 5    | {"context": retriever | format_docs, ...} | prompt   | answer(store, embeddings, llm, q, k) |
 * |      |   | llm | StrOutputParser()                        |                                      |
 * </pre>
 * Every method takes its models as parameters, so the same code runs with OpenAI in the demos and with fake models
 * in the offline tests. The class has no state: it is a toolbox, not an entry point.
 */
public final class RagPipeline {

    /** Identifies this program to web servers (the notebook warned that USER_AGENT was not set). */
    public static final String USER_AGENT = "BEPEC-langchaincommands/1.0 (Java; learning demo)";

    /** The notebook's RAG prompt: answer only from the retrieved context. */
    public static final String RAG_PROMPT = """

            Answer the question based only on the context below.

            Context: {context}

            Question: {question}

            Answer:""";

    /** Blocks whose content is never visible text: script, style, noscript, template and svg, with everything inside. */
    private static final java.util.regex.Pattern HIDDEN_BLOCKS = java.util.regex.Pattern.compile(
            "<(script|style|noscript|template|svg)\\b[^>]*>.*?</\\1\\s*>",
            java.util.regex.Pattern.CASE_INSENSITIVE | java.util.regex.Pattern.DOTALL);

    private RagPipeline() {
    }

    // ---------------------------------------------------------------- 1. LOAD

    /**
     * Step 1 - {@code WebBaseLoader(url).load()}: downloads a web page and converts it with {@link #fromHtml}.
     *
     * @param url the page to load
     * @return one Document for the page
     * @throws IOException if the page cannot be downloaded or the server answers with an error status
     */
    public static Document loadWebPage(String url) throws IOException, InterruptedException {
        HttpClient http = HttpClient.newBuilder()
                .followRedirects(HttpClient.Redirect.NORMAL)
                .connectTimeout(Duration.ofSeconds(20))
                .build();
        HttpRequest request = HttpRequest.newBuilder(URI.create(url))
                .header("User-Agent", USER_AGENT)
                .timeout(Duration.ofSeconds(30))
                .GET()
                .build();
        HttpResponse<String> response = http.send(request, HttpResponse.BodyHandlers.ofString());
        if (response.statusCode() / 100 != 2) {
            throw new IOException("HTTP " + response.statusCode() + " for " + url);
        }
        return fromHtml(response.body(), url);
    }

    /**
     * Step 1, offline part - turns HTML into a Document, like WebBaseLoader's BeautifulSoup step. It uses the HTML
     * parser built into the JDK ({@code javax.swing.text.html}), so no extra library is needed.
     * <ul>
     *   <li>text: the visible text of the page, title first - tags, scripts and styles removed, HTML entities
     *       decoded, whitespace collapsed;</li>
     *   <li>metadata: {@code source} (the URL), {@code title}, {@code description} (the page's meta description) and
     *       {@code language} (the html lang attribute). Values the page does not have are left out.</li>
     * </ul>
     * The metadata travels with the text, so answers can later cite where they came from.
     */
    public static Document fromHtml(String html, String url) {
        HtmlText page = new HtmlText();
        // remove invisible blocks first: the JDK parser follows HTML 3.2 rules and silently drops e.g. a <style> tag
        // inside <body> (common in modern pages), which would let the CSS through as if it were text
        String visibleHtml = HIDDEN_BLOCKS.matcher(html).replaceAll(" ");
        try {
            new ParserDelegator().parse(new StringReader(visibleHtml), page, true); // true = ignore the charset tag
        } catch (IOException e) {
            throw new UncheckedIOException("Could not parse the HTML of " + url, e); // cannot happen for a String
        }
        Metadata metadata = new Metadata().put("source", url);
        if (!page.title().isBlank()) {
            metadata.put("title", page.title());
        }
        if (!page.description.isBlank()) {
            metadata.put("description", page.description);
        }
        if (!page.language.isBlank()) {
            metadata.put("language", page.language);
        }
        return Document.from(page.text(), metadata);
    }

    /**
     * Collects what {@link #fromHtml} needs while the JDK parser walks through the page: the visible text, the title,
     * the meta description and the html lang attribute. Text inside script and style tags is skipped.
     */
    private static final class HtmlText extends HTMLEditorKit.ParserCallback {
        private final StringBuilder text = new StringBuilder();
        private final StringBuilder title = new StringBuilder();
        private String description = "";
        private String language = "";
        private int hiddenDepth;   // > 0 while inside script / style
        private boolean inTitle;

        /**
         * Tags whose content is never visible text. Matched by name, because the JDK parser follows the old HTML 3.2
         * rules and reports e.g. a {@code <style>} inside {@code <body>} (common in modern pages) as an unknown tag.
         */
        private static final java.util.Set<String> HIDDEN = java.util.Set.of("script", "style", "noscript", "template",
                "svg");

        private static boolean hidden(HTML.Tag tag) {
            return HIDDEN.contains(tag.toString().toLowerCase(java.util.Locale.ROOT));
        }

        @Override
        public void handleStartTag(HTML.Tag tag, MutableAttributeSet attributes, int pos) {
            if (hidden(tag)) {
                hiddenDepth++;
            } else if (tag == HTML.Tag.TITLE) {
                inTitle = true;
            } else if (tag == HTML.Tag.HTML) {
                language = attribute(attributes, HTML.Attribute.LANG);
            }
            text.append(' '); // a tag boundary separates words, e.g. </h1><p>
        }

        @Override
        public void handleEndTag(HTML.Tag tag, int pos) {
            if (hidden(tag) && hiddenDepth > 0) {
                hiddenDepth--;
            } else if (tag == HTML.Tag.TITLE) {
                inTitle = false;
            }
            text.append(' ');
        }

        @Override
        public void handleSimpleTag(HTML.Tag tag, MutableAttributeSet attributes, int pos) {
            if (tag == HTML.Tag.META && "description".equalsIgnoreCase(attribute(attributes, HTML.Attribute.NAME))) {
                description = attribute(attributes, HTML.Attribute.CONTENT);
            } else if (tag == HTML.Tag.HTML) {
                language = attribute(attributes, HTML.Attribute.LANG);
            }
            text.append(' ');
        }

        @Override
        public void handleText(char[] data, int pos) {
            if (hiddenDepth > 0) {
                return;
            }
            if (inTitle) {
                title.append(data);
            }
            text.append(data).append(' ');
        }

        String text() {
            return text.toString().replaceAll("\\s+", " ").strip();
        }

        String title() {
            return title.toString().replaceAll("\\s+", " ").strip();
        }

        private static String attribute(MutableAttributeSet attributes, HTML.Attribute name) {
            Object value = attributes.getAttribute(name);
            return value == null ? "" : value.toString().strip();
        }
    }

    // ---------------------------------------------------------------- 2. CHUNK

    /**
     * Step 2 - {@code RecursiveCharacterTextSplitter(chunk_size, chunk_overlap).split_text(text)}.
     * <p>
     * Cuts the text into chunks of at most {@code chunkSize} characters. It tries the biggest natural break first
     * (paragraph, then line, then sentence, then word) so chunks stay readable, and repeats about {@code overlap}
     * characters between neighbouring chunks so a sentence cut at a boundary is complete in at least one chunk.
     *
     * @param text      the text to split
     * @param chunkSize maximum characters per chunk (e.g. 200)
     * @param overlap   characters shared by neighbouring chunks (e.g. 30)
     * @return the chunks, in order
     */
    public static List<String> split(String text, int chunkSize, int overlap) {
        return splitDocument(Document.from(text), chunkSize, overlap).stream().map(TextSegment::text).toList();
    }

    /**
     * Step 2 for a loaded Document - like {@code splitter.split_documents([doc])}: same splitting as
     * {@link #split}, but every chunk keeps the document's metadata (source, title, ...), so search results can still
     * say where they came from.
     */
    public static List<TextSegment> splitDocument(Document document, int chunkSize, int overlap) {
        return DocumentSplitters.recursive(chunkSize, overlap).split(document);
    }

    // ---------------------------------------------------------------- 3. EMBED

    /**
     * Step 3 - {@code embeddings.embed_query(text)}: turns one text, typically the user's question, into a vector
     * (1536 numbers for text-embedding-3-small).
     */
    public static Embedding embedQuery(EmbeddingModel embeddings, String text) {
        return embeddings.embed(text).content();
    }

    /**
     * Step 3 - {@code embeddings.embed_documents(texts)}: turns many texts into vectors with one call, in the same
     * order as the input.
     */
    public static List<Embedding> embedDocuments(EmbeddingModel embeddings, List<String> texts) {
        return embeddings.embedAll(texts.stream().map(TextSegment::from).toList()).content();
    }

    /**
     * Step 3, the idea behind search - cosine similarity of two vectors: 1 = same direction (same meaning), around
     * 0 = unrelated. Vector stores rank search results by this kind of score.
     */
    public static double cosine(Embedding a, Embedding b) {
        float[] x = a.vector();
        float[] y = b.vector();
        double dot = 0, nx = 0, ny = 0;
        for (int i = 0; i < x.length; i++) {
            dot += x[i] * y[i];
            nx += x[i] * x[i];
            ny += y[i] * y[i];
        }
        return dot / (Math.sqrt(nx) * Math.sqrt(ny));
    }

    // ---------------------------------------------------------------- 4. STORE + SEARCH

    /**
     * Step 4 - {@code FAISS.from_documents(docs, embeddings)}: embeds plain texts in one call and stores
     * (vector, text) pairs in a new in-memory vector store.
     */
    public static InMemoryEmbeddingStore<TextSegment> fromDocuments(List<String> docs, EmbeddingModel embeddings) {
        return fromSegments(docs.stream().map(TextSegment::from).toList(), embeddings);
    }

    /**
     * Step 4 for chunks from {@link #splitDocument} - same as {@link #fromDocuments}, but keeps each chunk's metadata
     * in the store.
     */
    public static InMemoryEmbeddingStore<TextSegment> fromSegments(List<TextSegment> segments,
                                                                   EmbeddingModel embeddings) {
        InMemoryEmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        store.addAll(embeddings.embedAll(segments).content(), segments);
        return store;
    }

    /**
     * Step 4 - {@code vectorstore.similarity_search(query, k)}: embeds the query with the same model used for the
     * documents (vectors from different models cannot be compared) and returns the k most similar entries, most
     * similar first, each with its similarity score. This is a search by meaning, not by keywords.
     */
    public static List<EmbeddingMatch<TextSegment>> similaritySearch(InMemoryEmbeddingStore<TextSegment> store,
                                                                     EmbeddingModel embeddings, String query, int k) {
        return store.search(EmbeddingSearchRequest.builder()
                .queryEmbedding(embedQuery(embeddings, query))
                .maxResults(k)
                .build()).matches();
    }

    // ---------------------------------------------------------------- 5. ANSWER (the RAG chain)

    /** The notebook's {@code format_docs(docs)}: the retrieved texts joined with blank lines, ready for the prompt. */
    public static String formatDocs(List<EmbeddingMatch<TextSegment>> docs) {
        return docs.stream().map(m -> m.embedded().text()).collect(Collectors.joining("\n\n"));
    }

    /**
     * Step 5 - {@code rag.invoke(question)}, the full RAG chain:
     * <ol>
     *   <li><b>Retrieve</b> - the {@code k} chunks most similar to the question ({@link #similaritySearch}).</li>
     *   <li><b>Augment</b> - join them ({@link #formatDocs}) and put them into {@link #RAG_PROMPT} as the context,
     *       together with the question.</li>
     *   <li><b>Generate</b> - the model answers based only on that context.</li>
     * </ol>
     *
     * @param store      the vector store with the documents
     * @param embeddings the embedding model used to build the store
     * @param llm        the chat model that writes the answer
     * @param question   the user's question
     * @param k          how many chunks to retrieve
     * @return the answer text
     */
    public static String answer(InMemoryEmbeddingStore<TextSegment> store, EmbeddingModel embeddings, ChatModel llm,
                                String question, int k) {
        String context = formatDocs(similaritySearch(store, embeddings, question, k));
        return PromptChain.fromTemplate(RAG_PROMPT, llm).invoke(Map.of("context", context, "question", question));
    }
}
