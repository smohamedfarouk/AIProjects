package com.bepec.ragchunking.rag;

import com.bepec.ragchunking.retrieval.Retriever;
import dev.langchain4j.agent.tool.P;
import dev.langchain4j.agent.tool.Tool;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.service.AiServices;

/**
 * <b>Agentic RAG</b> - the notebook's {@code create_retriever_tool(...)} + {@code create_react_agent(llm, tools)}.
 * <p>
 * In basic RAG retrieval always happens exactly once. Here the retriever is a <i>tool</i> and the LLM decides:
 * <pre>
 *  prompt -> LLM -> "Shall I search for external data?"
 *                     yes -> call knowledge_base(query) -> retrieved info -> LLM (may search again) -> response
 *                     no  -> response
 * </pre>
 * LangChain4j's {@link AiServices} runs this ReAct loop: it offers the {@code knowledge_base} tool to the model,
 * executes every tool call the model makes, feeds the results back, and repeats until the model answers.
 */
public class KnowledgeBaseAgent {

    /** The agent's interface - AiServices generates the implementation. */
    interface Assistant {
        String chat(String userMessage);
    }

    /** {@code create_retriever_tool(retriever, name="knowledge_base", description=...)}. */
    public static class KnowledgeBaseTool {
        private final Retriever retriever;

        public KnowledgeBaseTool(Retriever retriever) {
            this.retriever = retriever;
        }

        @Tool(name = "knowledge_base", value = "Search docs about RAG, LangGraph, and vector databases.")
        public String search(@P("query to look up in the knowledge base") String query) {
            return RagChain.formatDocs(retriever.retrieve(query));
        }
    }

    private final Assistant assistant;

    public KnowledgeBaseAgent(ChatModel llm, Retriever retriever) {
        this.assistant = AiServices.builder(Assistant.class)
                .chatModel(llm)
                .tools(new KnowledgeBaseTool(retriever))
                .build();
    }

    /** {@code agent.invoke({"messages": [("user", question)]})["messages"][-1].content}. */
    public String invoke(String question) {
        return assistant.chat(question);
    }
}
