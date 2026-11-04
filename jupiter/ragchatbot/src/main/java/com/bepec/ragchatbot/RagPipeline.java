package com.bepec.ragchatbot;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * The RAG pipeline from the notebook: split -> embed -> store -> retrieve -> prompt -> answer.
 * <p>
 * Two phases:
 * <ul>
 *   <li><b>Indexing</b> (once, {@link #index}): pages are cut into chunks, every chunk is embedded, and the vectors
 *       are stored in an in-memory vector store.</li>
 *   <li><b>Answering</b> (per question, {@link #answer}): the question is embedded, the 4 most similar chunks are
 *       retrieved, and the model answers using only those chunks.</li>
 * </ul>
 */
public final class RagPipeline {

    /** RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150). */
    static final int CHUNK_SIZE = 1000;
    static final int CHUNK_OVERLAP = 150;

    /** search_kwargs={"k": 4}. */
    static final int TOP_K = 4;

    static final String SYSTEM_PROMPT =
            "You are a helpful assistant. Answer the question using ONLY the context below. "
                    + "If the answer is not in the context, say you don't know. "
                    + "Mention the page number(s) you used.\n\n"
                    + "Context:\n%s";

    private final EmbeddingModel embeddingModel;
    private final ChatModel llm;
    private final InMemoryEmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
    private final List<TextSegment> chunks = new ArrayList<>();
    private int dimension;

    public RagPipeline(EmbeddingModel embeddingModel, ChatModel llm) {
        this.embeddingModel = embeddingModel;
        this.llm = llm;
    }

    /**
     * Indexing - the Java version of {@code splitter.split_documents(pages)} plus
     * {@code Chroma.from_documents(documents=chunks, embedding=embeddings, collection_name="pdf_docs")}.
     * <ol>
     *   <li><b>Split</b> each page into chunks of at most 1000 characters with 150 characters of overlap. The
     *       recursive splitter tries to cut at paragraph breaks first, then lines, sentences, words - so chunks stay
     *       readable. The overlap repeats the end of one chunk at the start of the next, so a sentence cut at a
     *       boundary is still complete in at least one chunk. Each chunk keeps its page's metadata (page number).</li>
     *   <li><b>Embed</b> all chunks in one batch call - each becomes a vector of numbers that captures its
     *       meaning.</li>
     *   <li><b>Store</b> each vector together with its chunk in the in-memory vector store (the role Chroma plays
     *       in the notebook; it is also in-memory there, so both are rebuilt on every run).</li>
     * </ol>
     *
     * @param pages one document per PDF page (from {@link PdfLoader})
     * @return the number of chunks indexed
     */
    public int index(List<Document> pages) {
        DocumentSplitter splitter = DocumentSplitters.recursive(CHUNK_SIZE, CHUNK_OVERLAP);
        List<TextSegment> newChunks = new ArrayList<>();
        for (Document page : pages) {
            if (page.text() != null && !page.text().isBlank()) {
                newChunks.addAll(splitter.split(page));
            }
        }
        if (newChunks.isEmpty()) {
            throw new IllegalArgumentException("The PDF has no extractable text to index.");
        }

        List<Embedding> vectors = embeddingModel.embedAll(newChunks).content();
        store.addAll(vectors, newChunks);
        chunks.addAll(newChunks);
        dimension = vectors.getFirst().dimension();
        return newChunks.size();
    }

    /**
     * Retrieval - the Java version of {@code vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})}.
     * <ol>
     *   <li>Embed the question with the same embedding model used for the chunks (vectors from different models
     *       cannot be compared).</li>
     *   <li>Find the {@link #TOP_K} chunks whose vectors are most similar to the question's vector (cosine
     *       similarity: how closely the two vectors point in the same direction).</li>
     * </ol>
     * This is a <i>meaning</i> search, not a keyword search: "what do they teach in Python" can match a chunk that
     * says "Introduction to Python, Data Types, Loops" even without the word "teach".
     *
     * @param question the user's question
     * @return the best matches, most similar first, each with its similarity score and chunk
     */
    public List<EmbeddingMatch<TextSegment>> retrieve(String question) {
        Embedding query = embeddingModel.embedAll(List.of(TextSegment.from(question))).content().getFirst();
        return store.search(EmbeddingSearchRequest.builder()
                .queryEmbedding(query)
                .maxResults(TOP_K)
                .build()).matches();
    }

    /**
     * Java version of the notebook's {@code format_docs(docs)}: joins the retrieved chunks into one context string,
     * each preceded by {@code [Page n]} (page metadata + 1, i.e. the page number a human sees in the PDF). These
     * labels are what let the model "mention the page number(s) you used".
     *
     * @param retrieved chunks returned by {@link #retrieve}
     * @return the context text placed into the system prompt
     */
    static String formatDocs(List<TextSegment> retrieved) {
        return retrieved.stream()
                .map(chunk -> {
                    Integer page = chunk.metadata().getInteger("page");
                    return "[Page " + ((page == null ? 0 : page) + 1) + "]\n" + chunk.text();
                })
                .collect(Collectors.joining("\n\n"));
    }

    /**
     * Builds the two prompt messages - the Java version of the notebook's {@code ChatPromptTemplate}: a system
     * message with the rules plus the retrieved context, and a human message with the question.
     */
    static List<ChatMessage> buildPrompt(String context, String question) {
        return List.of(
                SystemMessage.from(SYSTEM_PROMPT.formatted(context)),
                UserMessage.from(question));
    }

    /**
     * Answers one question - the Java version of {@code rag_chain.invoke(question)}:
     * <pre>
     * {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
     * </pre>
     * <ol>
     *   <li><b>Retrieve</b> the 4 most relevant chunks ({@link #retrieve}).</li>
     *   <li><b>Format</b> them with page labels ({@link #formatDocs}).</li>
     *   <li><b>Prompt</b>: rules + context in the system message, the question as the user message
     *       ({@link #buildPrompt}).</li>
     *   <li><b>Generate</b>: gpt-4o-mini (temperature 0) answers from that context only, and says it doesn't know
     *       if the context does not contain the answer.</li>
     *   <li><b>Parse</b>: return just the reply text (StrOutputParser).</li>
     * </ol>
     * Each question is independent - there is no chat memory between questions.
     *
     * @param question the user's question
     * @return the model's answer, normally with the page numbers it used
     */
    public String answer(String question) {
        List<TextSegment> retrieved = retrieve(question).stream().map(EmbeddingMatch::embedded).toList();
        return llm.chat(buildPrompt(formatDocs(retrieved), question)).aiMessage().text();
    }

    /** Number of chunks in the store ({@code len(chunks)} in the notebook). */
    public int chunkCount() {
        return chunks.size();
    }

    /** All indexed chunks, in order. */
    public List<TextSegment> chunks() {
        return List.copyOf(chunks);
    }

    /** Length of each stored vector, e.g. 1536 for text-embedding-3-small (the notebook's {@code vector.shape}). */
    public int dimension() {
        return dimension;
    }
}
