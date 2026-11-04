package com.jupiter.langchaincommands.chat.messages;

import com.jupiter.langchaincommands.chat.chatprompttemplate.DraftEmailTemplateMain;
import com.jupiter.langchaincommands.chat.chatprompttemplate.MedicalTemplateMain;
import com.jupiter.langchaincommands.chat.selfcorrectingprompt.ReflectionMain;
import com.jupiter.langchaincommands.common.Models;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.response.ChatResponse;

/**
 * Requirement 2, step 1 - talking to the model with roles: SystemMessage + HumanMessage (notebook cells 10-15).
 * <b>Entry point.</b>
 * <p>
 * The prompt lessons, one entry point each: this class (roles) -> {@link DraftEmailTemplateMain} and
 * {@link MedicalTemplateMain} (reusable templates) -> {@link ReflectionMain} (self-correction).
 *
 * <h2>Run</h2>
 * <pre>
 *   mvn compile exec:java@messages
 * </pre>
 * In IntelliJ, run this class's {@code main}. Needs OPENAI_API_KEY (environment or .env). One paid OpenAI call.
 *
 * <h2>Messages and roles</h2>
 * A chat model does not receive one block of text: it receives a list of messages, each with a role.
 * <ul>
 *   <li><b>SystemMessage</b> - who the model is and how it should behave: "You are a Senior AI Corporate Trainer for
 *       Top MNCs". Written once by the developer; it shapes every answer.</li>
 *   <li><b>HumanMessage</b> (UserMessage in Java) - what the user asks: "Give me 3-Days Training Plan".</li>
 * </ul>
 * The same human question with a different system message gives a very different answer - e.g. a school teacher's
 * plan instead of a corporate one. The system message is the simplest way to steer the model.
 *
 * <h2>Notebook result</h2>
 * <pre>
 * | Cell | What it does                         | Result                                              |
 * |------|--------------------------------------|-----------------------------------------------------|
 * | 15   | System and human messages to         | Produced a full 3-day corporate AI training plan.   |
 * |      | gpt-4o-mini                          |                                                     |
 * </pre>
 */
public final class SystemHumanMessagesMain {

    static final String SYSTEM = "You are a Senior AI Corporate Trainer for Top MNCs";
    static final String QUESTION = "Give me 3-Days Training Plan";

    private SystemHumanMessagesMain() {
    }

    /**
     * Entry point - the Java version of
     * {@code llm.invoke([SystemMessage(content=...), HumanMessage(content=...)])} with
     * {@code ChatOpenAI(model="gpt-4o-mini")}.
     * <ol>
     *   <li>Switch the console to UTF-8 and stop with a clear message if OPENAI_API_KEY is missing
     *       ({@link Models#init()}).</li>
     *   <li>Create gpt-4o-mini.</li>
     *   <li>Send two messages in one call: the system message (the role) first, then the human message (the
     *       question).</li>
     *   <li>Print the reply text ({@code response.content} in Python).</li>
     * </ol>
     *
     * @param args not used
     */
    public static void main(String[] args) {
        Models.init();
        Models.section("SystemMessage + HumanMessage");
        ChatModel llm = Models.openAi("gpt-4o-mini");
        ChatResponse response = llm.chat(SystemMessage.from(SYSTEM), UserMessage.from(QUESTION));
        System.out.println(response.aiMessage().text());
    }
}
