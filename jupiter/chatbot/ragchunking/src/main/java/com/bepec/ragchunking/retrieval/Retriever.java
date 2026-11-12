package com.bepec.ragchunking.retrieval;

import dev.langchain4j.data.segment.TextSegment;

import java.util.List;

/** LangChain's {@code BaseRetriever}: {@code retriever.invoke(query)} returns the most relevant documents. */
@FunctionalInterface
public interface Retriever {

    List<TextSegment> retrieve(String query);
}
