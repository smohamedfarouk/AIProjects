package com.bepec.ragchunking;

import com.bepec.ragchunking.Fakes.FakeLlm;
import com.bepec.ragchunking.Fakes.WordEmbeddings;
import com.bepec.ragchunking.rag.CorrectiveRagGraph;
import com.bepec.ragchunking.rag.KnowledgeBaseAgent;
import com.bepec.ragchunking.rag.RagChain;
import com.bepec.ragchunking.retrieval.Bm25Retriever;
import com.bepec.ragchunking.retrieval.EnsembleRetriever;
import com.bepec.ragchunking.retrieval.MultiQueryRetriever;
import com.bepec.ragchunking.retrieval.RerankingRetriever;
import com.bepec.ragchunking.retrieval.Retriever;
import com.bepec.ragchunking.retrieval.VectorStore;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.scoring.ScoringModel;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class RagTypesTest {

    static final List<TextSegment> DOCS = SampleData.knowledgeBase();
    static final String RAG = DOCS.get(0).text();
    static final String LANGGRAPH = DOCS.get(1).text();
    static final String VECTOR_DBS = DOCS.get(2).text();

    static VectorStore store() {
        return VectorStore.fromDocuments(DOCS, new WordEmbeddings());
    }

    static List<String> texts(List<TextSegment> docs) {
        return docs.stream().map(TextSegment::text).toList();
    }

    @Test
    void basicRagPutsRetrievedContextIntoThePrompt() {
        FakeLlm llm = FakeLlm.answering("RAG combines retrieval with generation to reduce hallucinations.");
        String answer = new RagChain(store().asRetriever(2), llm).invoke("What is RAG?");

        assertEquals("RAG combines retrieval with generation to reduce hallucinations.", answer);
        assertTrue(llm.prompt(0).startsWith("Answer using only this context:\n" + RAG + "\n\n"), llm.prompt(0));
        assertTrue(llm.prompt(0).endsWith("\n\nQuestion: What is RAG?"));
    }

    @Test
    void bm25FindsExactKeywords() {
        Bm25Retriever bm25 = Bm25Retriever.fromDocuments(DOCS).setK(2);
        double[] scores = bm25.scores("LangGraph workflows");
        assertTrue(scores[1] > 0);
        assertEquals(0.0, scores[0]);
        assertEquals(LANGGRAPH, bm25.retrieve("LangGraph workflows").getFirst().text());
    }

    @Test
    void hybridSearchMatchesNotebook() {
        Retriever hybrid = new EnsembleRetriever(
                List.of(Bm25Retriever.fromDocuments(DOCS).setK(2), store().asRetriever(2)), List.of(0.4, 0.6));
        // both retrievers return the same two docs; duplicates are merged and LangGraph ranks first
        assertEquals(List.of(LANGGRAPH, VECTOR_DBS), texts(hybrid.retrieve("LangGraph workflows")));
    }

    @Test
    void multiQueryRetrievesForEachGeneratedQuestionAndDeduplicates() {
        FakeLlm llm = FakeLlm.answering("""
                What does RAG stand for?

                How does retrieval reduce hallucinations?
                What are vector databases used for?
                """);
        var retriever = MultiQueryRetriever.fromLlm(store().asRetriever(1), llm);

        assertEquals(3, retriever.generateQueries("What is RAG?").size());
        List<String> found = texts(retriever.retrieve("What is RAG?"));
        assertEquals(List.of(RAG, VECTOR_DBS), found);
        assertTrue(llm.prompt(0).endsWith("Original question: What is RAG?"));
    }

    @Test
    void rerankingKeepsTheTopNByRerankerScore() {
        ScoringModel reranker = (segments, query) ->
                Response.from(segments.stream().map(s -> s.text().contains("LangGraph") ? 0.9 : 0.1).toList());
        var retriever = new RerankingRetriever(store().asRetriever(10), reranker, 2);

        List<TextSegment> top = retriever.retrieve("What is RAG?");
        assertEquals(2, top.size());
        assertEquals(LANGGRAPH, top.getFirst().text());
        assertEquals(0.9, top.getFirst().metadata().getDouble("relevance_score"));
    }

    @Test
    void agentCallsTheKnowledgeBaseToolThenAnswers() {
        FakeLlm llm = new FakeLlm(request -> {
            if (request.messages().getLast() instanceof ToolExecutionResultMessage result) {
                return AiMessage.from("Answer using: " + result.text());
            }
            assertEquals("knowledge_base", request.toolSpecifications().getFirst().name());
            return AiMessage.from(ToolExecutionRequest.builder()
                    .id("call-1").name("knowledge_base").arguments("{\"query\": \"LangGraph workflows\"}").build());
        });

        String answer = new KnowledgeBaseAgent(llm, store().asRetriever(1))
                .invoke("Compare RAG and LangGraph using the knowledge base.");

        assertEquals("Answer using: " + LANGGRAPH, answer);
        assertEquals(2, llm.requests.size());
    }

    @Test
    void correctiveRagGeneratesDirectlyWhenDocsAreRelevant() {
        FakeLlm llm = new FakeLlm(r -> AiMessage.from(
                r.messages().getLast().toString().contains("Are docs relevant?") ? "Yes." : "RAG reduces hallucinations."));
        CorrectiveRagGraph.Result result = new CorrectiveRagGraph(store().asRetriever(2), llm).invoke("What is RAG?");

        assertEquals(List.of("retrieve", "grade", "generate"), result.path());
        assertEquals("RAG reduces hallucinations.", result.state().answer());
        assertFalse(result.state().documents().contains("[web search results here]"));
        assertTrue(llm.prompt(0).startsWith("Docs: ['" + RAG + "'"), llm.prompt(0));
    }

    @Test
    void correctiveRagFallsBackToWebSearchWhenDocsAreIrrelevant() {
        FakeLlm llm = new FakeLlm(r -> AiMessage.from(
                r.messages().getLast().toString().contains("Are docs relevant?") ? "no" : "from the web"));
        CorrectiveRagGraph.Result result = new CorrectiveRagGraph(store().asRetriever(2), llm).invoke("Who won?");

        assertEquals(List.of("retrieve", "grade", "web_search", "generate"), result.path());
        assertEquals("[web search results here]", result.state().documents().getLast());
        assertTrue(llm.prompt(1).contains("[web search results here]"));
        assertEquals("from the web", result.state().answer());
    }
}
