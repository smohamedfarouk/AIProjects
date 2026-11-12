package com.bepec.ragchunking;

import com.bepec.ragchunking.rag.CorrectiveRagGraph;
import com.bepec.ragchunking.rag.KnowledgeBaseAgent;
import com.bepec.ragchunking.rag.RagChain;
import com.bepec.ragchunking.retrieval.Bm25Retriever;
import com.bepec.ragchunking.retrieval.EnsembleRetriever;
import com.bepec.ragchunking.retrieval.MultiQueryRetriever;
import com.bepec.ragchunking.retrieval.RerankingRetriever;
import com.bepec.ragchunking.retrieval.Retriever;
import com.bepec.ragchunking.retrieval.VectorStore;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatModel;

import java.util.List;

/**
 * Part 2 of the notebook, "Types of RAG": the same three-document knowledge base queried with six RAG variants.
 *
 * <h2>Run</h2>
 * <pre>
 *   mvn compile exec:java -Dexec.mainClass=com.bepec.ragchunking.RagTypesMain
 * </pre>
 * Needs OPENAI_API_KEY. The re-ranking demo also needs COHERE_API_KEY and is skipped without it.
 *
 * <h2>Notebook cell -> Java</h2>
 * <pre>
 * | Notebook                                               | Java                                 |
 * |--------------------------------------------------------|--------------------------------------|
 * | Chroma.from_documents(docs, embeddings), k=2           | retrieval.VectorStore                |
 * | rag_chain (retriever, format_docs, prompt, llm, parser)| rag.RagChain                         |
 * | BM25Retriever + EnsembleRetriever([0.4, 0.6])          | retrieval.Bm25Retriever + Ensemble.. |
 * | MultiQueryRetriever.from_llm                           | retrieval.MultiQueryRetriever        |
 * | ContextualCompressionRetriever + CohereRerank(top_n=3) | retrieval.RerankingRetriever         |
 * | create_retriever_tool + create_react_agent             | rag.KnowledgeBaseAgent (AiServices)  |
 * | StateGraph: retrieve -> grade -> web_search/generate   | rag.CorrectiveRagGraph               |
 * </pre>
 */
public final class RagTypesMain {

    private RagTypesMain() {
    }

    public static void main(String[] args) {
        ChatModel llm = Config.llm();
        List<TextSegment> docs = SampleData.knowledgeBase();
        VectorStore vectorstore = VectorStore.fromDocuments(docs, Config.embeddings());
        Retriever retriever = vectorstore.asRetriever(2);

        heading("1. Basic RAG");
        System.out.println(new RagChain(retriever, llm).invoke("What is RAG?"));

        heading("2. Hybrid RAG (BM25 keyword 0.4 + vector semantic 0.6)");
        Retriever hybrid = new EnsembleRetriever(
                List.of(Bm25Retriever.fromDocuments(docs).setK(2), vectorstore.asRetriever(2)),
                List.of(0.4, 0.6));
        System.out.println(RagChain.formatDocs(hybrid.retrieve("LangGraph workflows")));

        heading("3. Multi-query RAG");
        var multiQuery = MultiQueryRetriever.fromLlm(vectorstore.asRetriever(2), llm);
        System.out.println("Retrieved " + multiQuery.retrieve("What is RAG?").size() + " unique docs");

        heading("4. Re-ranking RAG (retrieve 10, Cohere rerank, keep top 3)");
        if (Config.has("COHERE_API_KEY")) {
            var reranking = new RerankingRetriever(vectorstore.asRetriever(10), Config.reranker(), 3);
            System.out.println(new RagChain(reranking, llm).invoke("What is RAG?"));
        } else {
            System.out.println("COHERE_API_KEY not set - skipped.");
        }

        heading("5. Agentic RAG (the LLM decides when to search)");
        System.out.println(new KnowledgeBaseAgent(llm, retriever)
                .invoke("Compare RAG and LangGraph using the knowledge base."));

        heading("6. Corrective RAG (grade documents, fall back to web search)");
        CorrectiveRagGraph.Result result = new CorrectiveRagGraph(retriever, llm).invoke("What is RAG?");
        System.out.println("Path: " + String.join(" -> ", result.path()));
        System.out.println(result.state().answer());
    }

    private static void heading(String title) {
        System.out.println("\n=== " + title + " ===\n");
    }
}
