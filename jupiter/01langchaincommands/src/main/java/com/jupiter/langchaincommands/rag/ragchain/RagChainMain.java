package com.jupiter.langchaincommands.rag.ragchain;

import com.jupiter.langchaincommands.common.Models;
import com.jupiter.langchaincommands.rag.RagPipeline;
import com.jupiter.langchaincommands.rag.vectorstore.VectorStoreSearchMain;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.List;

/**
 * More LangChain commands, RAG lesson 5 - the full RAG chain: retrieve relevant documents, then answer from them
 * only (morelangchaincommands.ipynb cells 52-60). <b>Entry point.</b>
 * <p>
 * The RAG lessons: {@code rag.loader.WebLoaderMain} (load) -> {@code rag.chunking.ChunkingMain} (split) ->
 * {@code rag.embeddings.EmbeddingsMain} (embed) -> {@code rag.vectorstore.VectorStoreSearchMain} (store and search) -> this class.
 *
 * <h2>Run</h2>
 * <pre>
 *   mvn compile exec:java@rag-chain
 * </pre>
 * In IntelliJ, run this class's {@code main}. Needs OPENAI_API_KEY (environment or .env). Three small paid calls.
 *
 * <h2>RAG = Retrieval-Augmented Generation</h2>
 * The model knows nothing about BEPEC. Instead of hoping it guesses, we <b>retrieve</b> the most relevant facts from
 * our own documents and <b>augment</b> the prompt with them; the model then <b>generates</b> an answer based only on
 * that context.
 * <pre>
 * rag = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
 *
 *   question --retriever (k=2)--> 2 most similar documents --format_docs--> context text
 *   {context, question} --prompt--> "Answer ... based only on the context below. Context: ... Question: ..."
 *                       --llm-->    answer --StrOutputParser--> plain text
 * </pre>
 * "Based only on the context" keeps the model from inventing facts; if the documents don't contain the answer, the
 * model has nothing to base an invented one on.
 *
 * <h2>Notebook result</h2>
 * "What courses BEPEC offers" -> "BEPEC offers courses in Data Analytics, Data Engineering, AI/ML, and GenAI."
 */
public final class RagChainMain {

    static final List<String> DOCS = List.of(
            "BEPEC Solutions was founded 13+ years ago in Bangalore.",
            "BEPEC has placed 30,000+ alumni across 500+ hiring partners.",
            "BEPEC Offers this Courses which include Data Analytics, Data Engineering, AI/ML, GenAI.",
            "BEPEC is headquartered in Koramangala, Bengaluru.");

    static final String QUESTION = "What courses BEPEC offers";

    /** search_kwargs={"k": 2}. */
    static final int K = 2;

    private RagChainMain() {
    }

    /**
     * Entry point.
     * <ol>
     *   <li>Switch the console to UTF-8 and check OPENAI_API_KEY ({@link Models#init()}).</li>
     *   <li>Build a vector store from the four BEPEC facts (text-embedding-3-small).</li>
     *   <li>Print the 2 documents the retriever finds for the question, so you can see the context.</li>
     *   <li>Run the RAG chain with gpt-4o-mini ({@link RagPipeline#answer}) and print the answer.</li>
     * </ol>
     *
     * @param args not used
     */
    public static void main(String[] args) {
        Models.init();
        EmbeddingModel embeddings = Models.openAiEmbeddings("text-embedding-3-small");
        InMemoryEmbeddingStore<TextSegment> store = RagPipeline.fromDocuments(DOCS, embeddings);

        Models.section("retriever.invoke(\"" + QUESTION + "\") - the context");
        RagPipeline.similaritySearch(store, embeddings, QUESTION, K)
                .forEach(m -> System.out.println("- " + m.embedded().text()));

        Models.section("rag.invoke(\"" + QUESTION + "\")");
        System.out.println(RagPipeline.answer(store, embeddings, Models.openAi("gpt-4o-mini"), QUESTION, K));
    }
}
