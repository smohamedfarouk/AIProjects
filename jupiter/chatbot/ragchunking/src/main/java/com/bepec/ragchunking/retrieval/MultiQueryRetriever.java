package com.bepec.ragchunking.retrieval;

import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatModel;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * <b>Multi-query RAG</b> - a port of LangChain's {@code MultiQueryRetriever.from_llm(retriever, llm)}.
 * <p>
 * One phrasing of a question may miss relevant documents. So the LLM first rewrites the question into 3 different
 * versions, each version is searched separately, and the union of all results (duplicates removed) is returned.
 */
public class MultiQueryRetriever implements Retriever {

    /** LangChain's DEFAULT_QUERY_PROMPT. */
    static final String QUERY_PROMPT = """
            You are an AI language model assistant. Your task is
            to generate 3 different versions of the given user
            question to retrieve relevant documents from a vector database.
            By generating multiple perspectives on the user question,
            your goal is to help the user overcome some of the limitations
            of distance-based similarity search. Provide these alternative
            questions separated by newlines. Original question: %s""";

    private final Retriever retriever;
    private final ChatModel llm;

    public MultiQueryRetriever(Retriever retriever, ChatModel llm) {
        this.retriever = retriever;
        this.llm = llm;
    }

    /** {@code MultiQueryRetriever.from_llm(retriever=..., llm=...)}. */
    public static MultiQueryRetriever fromLlm(Retriever retriever, ChatModel llm) {
        return new MultiQueryRetriever(retriever, llm);
    }

    /** Asks the LLM for alternative questions: one per non-empty line of its reply. */
    public List<String> generateQueries(String question) {
        String reply = llm.chat(UserMessage.from(QUERY_PROMPT.formatted(question))).aiMessage().text();
        return reply.strip().lines().map(String::strip).filter(line -> !line.isEmpty()).toList();
    }

    @Override
    public List<TextSegment> retrieve(String query) {
        Set<TextSegment> unique = new LinkedHashSet<>();
        for (String q : generateQueries(query)) {
            unique.addAll(retriever.retrieve(q));
        }
        return new ArrayList<>(unique);
    }
}
