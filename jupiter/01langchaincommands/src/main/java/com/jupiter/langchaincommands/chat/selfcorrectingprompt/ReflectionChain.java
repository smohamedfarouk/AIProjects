package com.jupiter.langchaincommands.chat.selfcorrectingprompt;

import com.jupiter.langchaincommands.chat.chatprompttemplate.PromptChain;
import com.jupiter.langchaincommands.chat.chatprompttemplate.Templates;
import dev.langchain4j.model.chat.ChatModel;

import java.util.Map;

/**
 * Self-correction / reflection prompting: draft -> critique -> refine. The model reviews and improves its own
 * first answer, using three prompt templates in a row:
 * <ol>
 *   <li><b>draft</b> - write a short polite email declining the client's low-budget offer.</li>
 *   <li><b>critique</b> - find tone issues or rudeness in that draft (or say there are none).</li>
 *   <li><b>refine</b> - rewrite the draft using the critique; this final email is the result.</li>
 * </ol>
 * It costs three model calls instead of one, in exchange for a more polished answer. Java version of
 * <pre>
 * chain = ({"draft": draft_chain, "input": RunnablePassthrough()}
 *          | RunnablePassthrough.assign(critique = critique_chain)
 *          | refine_chain)
 * </pre>
 */
public final class ReflectionChain {

    /** Step 2 - the notebook's critique_prompt: the model reviews its own draft. */
    public static final String CRITIQUE_TEMPLATE =
            "Identify any tone issues or rudeness in this email draft, if it is perfect, say no isses. \n Draft: {draft}";

    /** Step 3 - the notebook's refine_prompt: rewrite the draft using the critique. */
    public static final String REFINE_TEMPLATE =
            "Rewrite the email based on the critique. \nOriginal Draft: {draft}\n Critique: {critique}\n Final Email:";

    private final PromptChain draftChain;
    private final PromptChain critiqueChain;
    private final PromptChain refineChain;

    /**
     * Creates the three prompt chains on one model: the notebook's draft_prompt (shared with the template lesson,
     * {@link Templates#DRAFT_TEMPLATE}), critique_prompt and refine_prompt.
     */
    public ReflectionChain(ChatModel llm) {
        this.draftChain = PromptChain.fromTemplate(Templates.DRAFT_TEMPLATE, llm);
        this.critiqueChain = PromptChain.fromTemplate(CRITIQUE_TEMPLATE, llm);
        this.refineChain = PromptChain.fromTemplate(REFINE_TEMPLATE, llm);
    }

    /**
     * Runs draft -> critique -> refine.
     * <ol>
     *   <li>draft = draft_chain(input) - Python: {@code {"draft": draft_chain, "input": RunnablePassthrough()}}.</li>
     *   <li>critique = critique_chain(draft) - Python: {@code RunnablePassthrough.assign(critique=critique_chain)},
     *       which keeps draft and input and adds critique.</li>
     *   <li>return refine_chain(draft, critique).</li>
     * </ol>
     *
     * @param input the client's offer, e.g. "The client offered 25000/day instead of 1lakh/day ..."
     * @return only the final, refined email
     */
    public String invoke(String input) {
        String draft = draftChain.invoke(input);                                   // {"draft": draft_chain, "input": passthrough}
        String critique = critiqueChain.invoke(Map.of("draft", draft, "input", input)); // .assign(critique=critique_chain)
        return refineChain.invoke(Map.of("draft", draft, "input", input, "critique", critique));
    }
}
