package com.jupiter.langchaincommands.rag.chunking;

import com.jupiter.langchaincommands.common.Models;
import com.jupiter.langchaincommands.rag.RagPipeline;

import java.util.List;

/**
 * More LangChain commands, RAG lesson 2 - chunking: splitting long text into small, overlapping pieces
 * (morelangchaincommands.ipynb cells 29-37). <b>Entry point.</b>
 * <p>
 * The RAG lessons: {@code rag.loader.WebLoaderMain} (load) -> this class (split) -> {@code rag.embeddings.EmbeddingsMain}
 * (embed) -> {@code rag.vectorstore.VectorStoreSearchMain} (store and search) -> {@code rag.ragchain.RagChainMain}.
 *
 * <h2>Run</h2>
 * <pre>
 *   mvn compile exec:java@chunking
 * </pre>
 * In IntelliJ, run this class's {@code main}. No API key needed - splitting happens locally.
 *
 * <h2>Why chunk</h2>
 * A whole document is too big to send to the model with every question, and one long text gets a single, blurry
 * embedding. So documents are cut into chunks: each chunk is embedded and searched on its own, and only the few
 * relevant chunks are given to the model.
 * <ul>
 *   <li><b>chunk_size = 200</b> - at most 200 characters per chunk.</li>
 *   <li><b>chunk_overlap = 30</b> - the end of one chunk is repeated at the start of the next, so a sentence cut at a
 *       boundary is still complete in at least one chunk.</li>
 *   <li><b>Recursive</b> - it tries to cut at the biggest natural break first (paragraph, then line, then sentence,
 *       then word), so chunks stay readable.</li>
 * </ul>
 * The notebook passed its own separator list ({@code "\n\n", "\n", ".", " "}); LangChain4j's recursive splitter uses
 * the same idea with its fixed order (paragraph, line, sentence, word), so the chunk boundaries - and the count, 13 in
 * the notebook - can differ slightly.
 */
public final class ChunkingMain {

    static final int CHUNK_SIZE = 200;
    static final int CHUNK_OVERLAP = 30;

    /** The notebook's sample paragraph, repeated 5 times to make a "long" text. */
    static final String LONG_TEXT = """

                LangChain is a framework for building applications powered by language models.
                It provides modular components like prompts, chains, agents, memory, and retrievers.
                With LangChain, you can build chatbots, RAG systems, AI agents, and much more.
                The framework supports multiple LLM providers including OpenAI, Anthropic, Google,
                and open-source models via Hugging Face and Ollama.
                """.repeat(5);

    private ChunkingMain() {
    }

    /**
     * Entry point.
     * <ol>
     *   <li>Switch the console to UTF-8 ({@link Models#initConsole()}) - no API key is needed.</li>
     *   <li>Split {@link #LONG_TEXT} ({@link RagPipeline#split}) into chunks of at most 200 characters with 30
     *       characters of overlap.</li>
     *   <li>Print the number of chunks and every chunk with its index and length, like the notebook's
     *       {@code for i, chunk in enumerate(chunks): print(i, chunk)}.</li>
     * </ol>
     *
     * @param args not used
     */
    public static void main(String[] args) {
        Models.initConsole();
        Models.section("RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)");
        List<String> chunks = RagPipeline.split(LONG_TEXT, CHUNK_SIZE, CHUNK_OVERLAP);
        System.out.println("Original text: " + LONG_TEXT.length() + " characters -> " + chunks.size() + " chunks\n");
        for (int i = 0; i < chunks.size(); i++) {
            System.out.println(i + " (" + chunks.get(i).length() + " chars) " + chunks.get(i));
        }
    }
}
