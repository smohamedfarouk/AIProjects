package com.bepec.ragchunking.rag;

import com.bepec.ragchunking.retrieval.Retriever;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatModel;

import java.util.List;
import java.util.stream.Collectors;

/**
 * <b>Basic (naive) RAG</b> - the notebook's LCEL chain:
 * <pre>
 * prompt = ChatPromptTemplate.from_template("Answer using only this context:\n{context}\n\nQuestion: {question}")
 * rag_chain = {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
 * </pre>
 * Prompt template + prompt + external data -> LLM -> response: retrieve documents for the question, paste them
 * into the prompt as context, and let the LLM answer from that context only. Any {@link Retriever} can be plugged
 * in - vector, hybrid, multi-query or re-ranking - which is how the notebook builds its other RAG types.
 */
public class RagChain {

    public static final String PROMPT = "Answer using only this context:\n%s\n\nQuestion: %s";

    private final Retriever retriever;
    private final ChatModel llm;

    public RagChain(Retriever retriever, ChatModel llm) {
        this.retriever = retriever;
        this.llm = llm;
    }

    /** {@code format_docs(docs)}: the documents' text separated by blank lines. */
    public static String formatDocs(List<TextSegment> docs) {
        return docs.stream().map(TextSegment::text).collect(Collectors.joining("\n\n"));
    }

    /** {@code rag_chain.invoke(question)}. */
    public String invoke(String question) {
        String context = formatDocs(retriever.retrieve(question));
        return llm.chat(UserMessage.from(PROMPT.formatted(context, question))).aiMessage().text();
    }
}
