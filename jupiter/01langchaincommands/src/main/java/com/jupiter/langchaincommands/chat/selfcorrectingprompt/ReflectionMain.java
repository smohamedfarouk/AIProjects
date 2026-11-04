package com.jupiter.langchaincommands.chat.selfcorrectingprompt;

import com.jupiter.langchaincommands.chat.messages.SystemHumanMessagesMain;
import com.jupiter.langchaincommands.chat.chatprompttemplate.DraftEmailTemplateMain;
import com.jupiter.langchaincommands.chat.chatprompttemplate.MedicalTemplateMain;
import com.jupiter.langchaincommands.chat.chatprompttemplate.Templates;
import com.jupiter.langchaincommands.common.Models;

/**
 * Requirement 2, step 3 - self-correction / reflection prompting: draft -> critique -> refine (notebook cells
 * 24-30). <b>Entry point.</b>
 * <p>
 * The prompt lessons, one entry point each: {@link SystemHumanMessagesMain} (roles) -> {@link DraftEmailTemplateMain}
 * and {@link MedicalTemplateMain} (reusable templates) -> this class (self-correction).
 *
 * <h2>Run</h2>
 * <pre>
 *   mvn compile exec:java@reflection
 * </pre>
 * In IntelliJ, run this class's {@code main}. Needs OPENAI_API_KEY (environment or .env). Three paid OpenAI calls.
 *
 * <h2>Why reflection</h2>
 * One prompt rarely gives the best result. Reflection chains three prompt templates so the model checks and improves
 * its own work:
 * <pre>
 *   offer --draft_chain-->    draft email
 *         --critique_chain--> "tone issues: ..."         (the model reviews its own draft)
 *         --refine_chain-->   improved final email       (rewritten using the critique)
 * </pre>
 * In Python this uses RunnablePassthrough to carry the original input and the draft along to the later steps;
 * {@link ReflectionChain} does the same with ordinary variables. It costs three model calls instead of one, in
 * exchange for a more polished answer. Compare the result with the single draft printed by
 * {@link DraftEmailTemplateMain} - same offer, same draft template.
 *
 * <h2>Notebook result</h2>
 * <pre>
 * | Cell | What it does                       | Result                                                             |
 * |------|------------------------------------|--------------------------------------------------------------------|
 * | 30   | Draft -> critique -> refine chain  | Rewrote the email in a warmer tone and offered to adjust the       |
 * |      |                                    | training scope to fit their budget.                                |
 * </pre>
 */
public final class ReflectionMain {

    private ReflectionMain() {
    }

    /**
     * Entry point - the Java version of
     * <pre>
     * chain = ({"draft": draft_chain, "input": RunnablePassthrough()}
     *          | RunnablePassthrough.assign(critique=critique_chain)
     *          | refine_chain)
     * print(chain.invoke("The client offered 25000/day instead of 1lakh/day for AI Corporate Training"))
     * </pre>
     * <ol>
     *   <li>Switch the console to UTF-8 and stop with a clear message if OPENAI_API_KEY is missing
     *       ({@link Models#init()}).</li>
     *   <li>Build the three chains on gpt-4o-mini ({@link ReflectionChain}).</li>
     *   <li>Run draft -> critique -> refine on the client's offer and print only the final, refined email.</li>
     * </ol>
     *
     * @param args not used
     */
    public static void main(String[] args) {
        Models.init();
        // Self-Correction/Reflection Prompting
        Models.section("draft -> critique -> refine chain");
        ReflectionChain reflection = new ReflectionChain(Models.openAi("gpt-4o-mini"));
        System.out.println(reflection.invoke(Templates.CLIENT_OFFER_INPUT_REASON));
    }
}
