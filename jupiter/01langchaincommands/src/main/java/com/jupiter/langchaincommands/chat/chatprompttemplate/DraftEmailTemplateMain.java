package com.jupiter.langchaincommands.chat.chatprompttemplate;

import com.jupiter.langchaincommands.chat.messages.SystemHumanMessagesMain;
import com.jupiter.langchaincommands.chat.selfcorrectingprompt.ReflectionMain;
import com.jupiter.langchaincommands.common.Models;

import java.util.Map;

/**
 * Requirement 2, step 2a - a reusable prompt: the draft-email ChatPromptTemplate run as an LCEL chain
 * {@code prompt | llm | parser} (notebook cells 16-20). <b>Entry point.</b>
 * <p>
 * The prompt lessons, one entry point each: {@link SystemHumanMessagesMain} (roles) -> this class and
 * {@link MedicalTemplateMain} (reusable templates) -> {@link ReflectionMain} (self-correction).
 *
 * <h2>Run</h2>
 * <pre>
 *   mvn compile exec:java@draft-template
 * </pre>
 * In IntelliJ, run this class's {@code main}. Needs OPENAI_API_KEY (environment or .env). One paid OpenAI call.
 *
 * <h2>ChatPromptTemplate: write the prompt once, fill it many times</h2>
 * Instead of writing a new prompt for every client, write it once with a {@code {placeholder}} and fill it per call:
 * <pre>
 *   template: "Draft a short polite email to a client declining their Corporate Training Proposal due to Low Budget: {input}"
 *   values:   {"input": "The client offered 25000/day instead of 1lakh/day for AI Corporate Training"}
 *   prompt:   "Draft a short polite email ... due to Low Budget: The client offered 25000/day instead of 1lakh/day ..."
 * </pre>
 * The same template works for any client offer - only the value changes.
 *
 * <h2>LCEL chains: prompt | llm | parser</h2>
 * LCEL (LangChain Expression Language) connects steps with {@code |}; each step's output feeds the next:
 * <pre>
 *   {"input": offer} -> ChatPromptTemplate (fill) -> llm (answer) -> StrOutputParser (plain text)
 * </pre>
 * {@link PromptChain} is that whole pipeline in one Java object: {@code PromptChain.fromTemplate(t, llm).invoke(values)}.
 * <p>
 * This is the same draft template the reflection lesson starts from, so you can compare this single draft with the
 * improved email from {@link ReflectionMain}.
 *
 * <h2>Notebook result</h2>
 * <pre>
 * | Cell | What it does    | Result                                                                   |
 * |------|-----------------|--------------------------------------------------------------------------|
 * | 20   | draft_chain     | Wrote a polite email declining ₹25,000/day against your usual             |
 * |      |                 | ₹1,00,000/day.                                                           |
 * </pre>
 */
public final class DraftEmailTemplateMain {

    private DraftEmailTemplateMain() {
    }

    /**
     * Entry point - the Java version of
     * <pre>
     * draft_prompt = ChatPromptTemplate.from_template("Draft a short polite email ...: {input}")
     * draft_chain = draft_prompt | llm | StrOutputParser()
     * print(draft_chain.invoke({"input": "The client offered 25000/day instead of 1lakh/day for AI Corporate Training"}))
     * </pre>
     * <ol>
     *   <li>Switch the console to UTF-8 and stop with a clear message if OPENAI_API_KEY is missing
     *       ({@link Models#init()}).</li>
     *   <li>Build the chain from {@link Templates#DRAFT_TEMPLATE} on gpt-4o-mini.</li>
     *   <li>Fill {@code {input}} with {@link Templates#CLIENT_OFFER_INPUT_REASON}, send it, and print the decline email.</li>
     * </ol>
     *
     * @param args not used
     */
    public static void main(String[] args) {
        Models.init();
        Models.section("draft_chain = draft_prompt | llm | StrOutputParser()");
        PromptChain draftChain = PromptChain.fromTemplate(Templates.DRAFT_TEMPLATE, Models.openAi("gpt-4o-mini"));
        System.out.println(draftChain.invoke(Map.of("input", Templates.CLIENT_OFFER_INPUT_REASON)));
    }
}
