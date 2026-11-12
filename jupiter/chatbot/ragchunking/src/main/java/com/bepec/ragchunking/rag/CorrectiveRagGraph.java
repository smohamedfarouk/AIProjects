package com.bepec.ragchunking.rag;

import com.bepec.ragchunking.Py;
import com.bepec.ragchunking.retrieval.Retriever;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatModel;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * <b>Corrective RAG (CRAG)</b> - the notebook's LangGraph {@code StateGraph}, as a small state machine.
 * <p>
 * Before answering, an LLM grades whether the retrieved documents are actually relevant; if not, the graph falls
 * back to a web search to get better context:
 * <pre>
 *   retrieve -> grade --(RELEVANT)--> generate -> END
 *                     \-(NEEDS_WEB)-> web_search -> generate -> END
 * </pre>
 * Each node reads the shared {@link State} and returns an updated copy - exactly like LangGraph nodes returning a
 * partial state dict.
 */
public class CorrectiveRagGraph {

    /** {@code class State(TypedDict): question, documents, answer}. */
    public record State(String question, List<String> documents, String answer) {
        State withDocuments(List<String> docs) {
            return new State(question, List.copyOf(docs), answer);
        }

        State withAnswer(String a) {
            return new State(question, documents, a);
        }
    }

    /** The final state plus the nodes visited, in order. */
    public record Result(State state, List<String> path) {
    }

    public static final String END = "END";
    static final String GRADE_PROMPT = "Docs: %s\nQuestion: %s\nAre docs relevant? Answer 'yes' or 'no'.";
    static final String GENERATE_PROMPT = "Context: %s\nQuestion: %s\nAnswer:";

    private final Retriever retriever;
    private final ChatModel llm;

    public CorrectiveRagGraph(Retriever retriever, ChatModel llm) {
        this.retriever = retriever;
        this.llm = llm;
    }

    /** {@code app.invoke({"question": question})}: runs from the entry point "retrieve" until END. */
    public Result invoke(String question) {
        State state = new State(question, List.of(), null);
        List<String> path = new ArrayList<>();
        String node = "retrieve";
        while (!END.equals(node)) {
            path.add(node);
            state = switch (node) {
                case "retrieve" -> retrieve(state);
                case "grade" -> grade(state);
                case "web_search" -> webSearch(state);
                case "generate" -> generate(state);
                default -> throw new IllegalStateException("Unknown node " + node);
            };
            node = switch (node) {                 // the edges
                case "retrieve" -> "grade";
                case "grade" -> decide(state);     // conditional edge
                case "web_search" -> "generate";
                default -> END;                    // generate -> END
            };
        }
        return new Result(state, List.copyOf(path));
    }

    /** Node: fetch documents for the question. */
    State retrieve(State state) {
        return state.withDocuments(retriever.retrieve(state.question()).stream().map(TextSegment::text).toList());
    }

    /** Node: the LLM grades whether the docs are relevant. */
    State grade(State state) {
        String verdict = ask(GRADE_PROMPT.formatted(Py.repr(state.documents()), state.question()));
        return state.withAnswer(verdict.toLowerCase(Locale.ROOT).contains("yes") ? "RELEVANT" : "NEEDS_WEB");
    }

    /** Conditional edge: which node runs after grading. */
    static String decide(State state) {
        return "RELEVANT".equals(state.answer()) ? "generate" : "web_search";
    }

    /** Node: fallback. In production call a real search API (Tavily, etc.). */
    State webSearch(State state) {
        List<String> docs = new ArrayList<>(state.documents());
        docs.add("[web search results here]");
        return state.withDocuments(docs);
    }

    /** Node: answer from the (possibly corrected) documents. */
    State generate(State state) {
        return state.withAnswer(ask(GENERATE_PROMPT.formatted(Py.repr(state.documents()), state.question())));
    }

    private String ask(String prompt) {
        return llm.chat(UserMessage.from(prompt)).aiMessage().text();
    }
}
