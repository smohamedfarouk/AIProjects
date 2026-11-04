package com.bepec.ragpinecone;

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
import dev.langchain4j.store.embedding.EmbeddingStore;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.stream.Collectors;

/**
 * The RAG pipeline of pdf_rag.ipynb, backed by Pinecone: split -> id -> embed -> upsert -> retrieve -> answer.
 * <p>
 * The vector store is passed in, so the same code runs against Pinecone in production and against an in-memory
 * store in the offline tests.
 */
public final class PineconeRagPipeline {

    /** RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150). */
    static final int CHUNK_SIZE = 1000;
    static final int CHUNK_OVERLAP = 150;

    /** vector_store.add_documents(..., batch_size=100). */
    static final int BATCH_SIZE = 100;

    /** search_kwargs={"k": 3}. */
    static final int TOP_K = 3;

    static final String SYSTEM_PROMPT =
            "You are a helpful assistant. Answer the question using ONLY the context below. "
                    + "If the answer is not in the context, say you don't know. "
                    + "Mention the page number(s) you used.\n\n"
                    + "Context:\n%s";

    private final EmbeddingModel embeddingModel;
    private final EmbeddingStore<TextSegment> store;
    private final ChatModel llm;

    public PineconeRagPipeline(EmbeddingModel embeddingModel, EmbeddingStore<TextSegment> store, ChatModel llm) {
        this.embeddingModel = embeddingModel;
        this.store = store;
        this.llm = llm;
    }

    /**
     * Splits PDF pages into chunks - {@code splitter.split_documents(pages)}.
     * <p>
     * Chunks are at most 1000 characters with 150 characters of overlap. The recursive splitter cuts at paragraph
     * breaks first, then lines, sentences and words, so chunks stay readable; the overlap means a sentence cut at a
     * boundary is complete in at least one chunk. Every chunk keeps its page's metadata (source file, page number).
     * Pages without text produce no chunks.
     *
     * @param pages one document per PDF page
     * @return all chunks, in page order
     */
    static List<TextSegment> split(List<Document> pages) {
        DocumentSplitter splitter = DocumentSplitters.recursive(CHUNK_SIZE, CHUNK_OVERLAP);
        List<TextSegment> chunks = new ArrayList<>();
        for (Document page : pages) {
            if (page.text() != null && !page.text().isBlank()) {
                chunks.addAll(splitter.split(page));
            }
        }
        return chunks;
    }

    /**
     * The chunk's id in Pinecone - the notebook's
     * {@code hashlib.md5(f"{c.metadata['source']}::{c.page_content}".encode()).hexdigest()}.
     * <p>
     * Why a hash instead of a random id: the same chunk text from the same file always gets the same id. Pinecone
     * "upserts" (insert or overwrite by id), so running the indexing again replaces the existing vectors instead of
     * adding duplicates - the index stays clean however often you run it. The formula is identical to Python's, so
     * Java and the notebook produce the same ids for the same chunk.
     * <p>
     * Side effect (same as in the notebook): the page number is not part of the id, so text repeated word-for-word
     * in one file (e.g. an identical header on several pages) is stored once, with the page of the last copy.
     *
     * @param chunk a chunk with a "source" metadata entry
     * @return 32-character lowercase hex MD5
     */
    static String chunkId(TextSegment chunk) {
        String source = chunk.metadata().getString("source");
        String key = (source == null ? "" : source) + "::" + chunk.text();
        try {
            byte[] digest = MessageDigest.getInstance("MD5").digest(key.getBytes(StandardCharsets.UTF_8));
            return HexFormat.of().formatHex(digest);
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("MD5 not available", e); // every JDK has MD5
        }
    }

    /**
     * Indexing - the Java version of {@code vector_store.add_documents(chunks, ids=ids, batch_size=100)}.
     * <ol>
     *   <li>Split the pages into chunks ({@link #split}) and compute each chunk's stable id ({@link #chunkId}).</li>
     *   <li>Work in batches of 100 chunks: embed the batch with one OpenAI call, then upsert the vectors, ids and
     *       chunks (text stored in metadata) into Pinecone with one request. Batching keeps each request small.</li>
     * </ol>
     * Because ids are content hashes, re-running this updates the same vectors instead of duplicating them. Note:
     * it only adds or overwrites - chunks that disappeared from a changed PDF stay in the index until removed.
     *
     * @param pages one document per PDF page
     * @return the number of chunks written
     */
    public int index(List<Document> pages) {
        List<TextSegment> chunks = split(pages);
        if (chunks.isEmpty()) {
            throw new IllegalArgumentException("The PDF has no extractable text to index.");
        }
        for (int from = 0; from < chunks.size(); from += BATCH_SIZE) {
            List<TextSegment> batch = chunks.subList(from, Math.min(from + BATCH_SIZE, chunks.size()));
            List<String> ids = batch.stream().map(PineconeRagPipeline::chunkId).toList();
            List<Embedding> vectors = embeddingModel.embedAll(batch).content();
            store.addAll(ids, vectors, batch);
        }
        return chunks.size();
    }

    /**
     * Retrieval - {@code vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})}.
     * <ol>
     *   <li>Embed the question with the same model used for the chunks.</li>
     *   <li>Ask Pinecone for the 3 vectors closest to it (cosine similarity) within this namespace.</li>
     * </ol>
     * It is a meaning search, not a keyword search, and it searches whatever is stored in the index - including
     * chunks indexed in earlier runs or by the Python notebook.
     *
     * @param question the user's question
     * @return the best matches, most similar first
     */
    public List<EmbeddingMatch<TextSegment>> retrieve(String question) {
        Embedding query = embeddingModel.embedAll(List.of(TextSegment.from(question))).content().getFirst();
        return store.search(EmbeddingSearchRequest.builder()
                .queryEmbedding(query)
                .maxResults(TOP_K)
                .build()).matches();
    }

    /**
     * The notebook's {@code format_docs(docs)}: joins the chunks into one context string, each preceded by
     * {@code [Page n]} so the model can cite pages. PyPDFLoader pages start at 0, so n = page + 1.
     * <p>
     * The page is read as any number: Pinecone returns numeric metadata as decimals (0.0), especially for vectors
     * written by the Python notebook.
     */
    static String formatDocs(List<TextSegment> chunks) {
        return chunks.stream()
                .map(chunk -> "[Page " + (pageOf(chunk) + 1) + "]\n" + chunk.text())
                .collect(Collectors.joining("\n\n"));
    }

    private static int pageOf(TextSegment chunk) {
        Object page = chunk.metadata().toMap().get("page");
        if (page instanceof Number n) {
            return n.intValue();
        }
        if (page instanceof String s && !s.isBlank()) {
            return (int) Double.parseDouble(s);
        }
        return 0;
    }

    /** The notebook's ChatPromptTemplate: rules + retrieved context as the system message, the question as human. */
    static List<ChatMessage> buildPrompt(String context, String question) {
        return List.of(SystemMessage.from(SYSTEM_PROMPT.formatted(context)), UserMessage.from(question));
    }

    /**
     * Answers one question - {@code rag_chain.invoke(question)}:
     * retrieve 3 chunks -> format with page labels -> prompt -> gpt-4o-mini (temperature 0) -> reply text.
     * The model is told to use only that context, say "I don't know" otherwise, and name the pages it used.
     * Questions are independent - there is no chat memory.
     */
    public String answer(String question) {
        List<TextSegment> retrieved = retrieve(question).stream().map(EmbeddingMatch::embedded).toList();
        return llm.chat(buildPrompt(formatDocs(retrieved), question)).aiMessage().text();
    }
}
