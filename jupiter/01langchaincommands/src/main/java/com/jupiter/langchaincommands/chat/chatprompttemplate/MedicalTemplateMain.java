package com.jupiter.langchaincommands.chat.chatprompttemplate;

import com.jupiter.langchaincommands.chat.messages.SystemHumanMessagesMain;
import com.jupiter.langchaincommands.chat.selfcorrectingprompt.ReflectionMain;
import com.jupiter.langchaincommands.common.Models;

import java.util.Map;

/**
 * Requirement 2, step 2b - a reusable prompt: the medicine ChatPromptTemplate run as an LCEL chain
 * {@code prompt | llm | parser} (notebook cells 21-23). <b>Entry point.</b>
 * <p>
 * The prompt lessons, one entry point each: {@link SystemHumanMessagesMain} (roles) -> {@link DraftEmailTemplateMain}
 * and this class (reusable templates) -> {@link ReflectionMain} (self-correction).
 *
 * <h2>Run</h2>
 * <pre>
 *   mvn compile exec:java@medical-template
 * </pre>
 * In IntelliJ, run this class's {@code main}. Needs OPENAI_API_KEY (environment or .env). One paid OpenAI call.
 *
 * <h2>Same technique, different job</h2>
 * {@link DraftEmailTemplateMain} uses a template to write emails; this class uses the very same mechanism for a
 * completely different task - summarising a medicine - just by writing a different template:
 * <pre>
 *   template: "Explain about this medicine Pros and Cons and side effects in 2Lines ...: {Name}"
 *   values:   {"Name": "Creatine"}
 *   prompt:   "Explain about this medicine Pros and Cons and side effects in 2Lines ...: Creatine"
 * </pre>
 * The template fixes the format every time (2 lines, pros/cons/side effects, the BEPEC sign-off); only the medicine
 * name changes. Change "Creatine" to any other medicine and the answer keeps the same shape.
 *
 * <h2>LCEL chain</h2>
 * <pre>
 *   {"Name": "Creatine"} -> ChatPromptTemplate (fill) -> llm (answer) -> StrOutputParser (plain text)
 * </pre>
 * {@link PromptChain} is that whole pipeline in one Java object.
 *
 * <h2>Notebook result</h2>
 * <pre>
 * | Cell | What it does                 | Result                                                             |
 * |------|------------------------------|--------------------------------------------------------------------|
 * | 23   | medical_chain ("Creatine")   | Two lines on pros, cons and side effects, ending with the          |
 * |      |                              | www.bepec.in sign-off.                                             |
 * </pre>
 */
public final class MedicalTemplateMain {

    /** The medicine to explain - change it to try the same template on another medicine. */
    static final String MEDICINE = "Creatine";

    private MedicalTemplateMain() {
    }

    /**
     * Entry point - the Java version of
     * <pre>
     * medical_prompt = ChatPromptTemplate.from_template("Explain about this medicine ...: {Name}")
     * medical_chain = medical_prompt | llm | StrOutputParser()
     * print(medical_chain.invoke({"Name": "Creatine"}))
     * </pre>
     * <ol>
     *   <li>Switch the console to UTF-8 and stop with a clear message if OPENAI_API_KEY is missing
     *       ({@link Models#init()}).</li>
     *   <li>Build the chain from {@link Templates#MEDICAL_TEMPLATE} on gpt-4o-mini.</li>
     *   <li>Fill {@code {Name}} with {@link #MEDICINE}, send it, and print the 2-line summary.</li>
     * </ol>
     *
     * @param args not used
     */
    public static void main(String[] args) {
        Models.init();
        Models.section("medical_chain = medical_prompt | llm | StrOutputParser()");
        PromptChain medicalChain = PromptChain.fromTemplate(Templates.MEDICAL_TEMPLATE, Models.openAi("gpt-4o-mini"));
        System.out.println(medicalChain.invoke(Map.of("Name", MEDICINE)));
    }
}
