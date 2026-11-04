package com.jupiter.langchaincommands.chat.chatprompttemplate;

import com.jupiter.langchaincommands.common.Models;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.model.chat.ChatModel;

import java.util.List;
import java.util.Map;

/**
 * More LangChain commands, lesson 2 - a prompt template with a system <b>and</b> a human message
 * (morelangchaincommands.ipynb cells 3-7). <b>Entry point.</b>
 * <p>
 * Builds on {@link DraftEmailTemplateMain} (one template, one user message): here each role has its own template.
 *
 * <h2>Run</h2>
 * <pre>
 *   mvn compile exec:java@messages-template
 * </pre>
 * In IntelliJ, run this class's {@code main}. Needs OPENAI_API_KEY (environment or .env). One paid OpenAI call.
 *
 * <h2>How it works</h2>
 * <pre>
 *   system template: "You are an expert in {domain}"                 -> "You are an expert in Banking"
 *   human template:  "Explain {topic} to a {audience} in 3 lines"    -> "Explain AWS BedRock to a Senior Architects in 3 lines"
 * </pre>
 * {@code format_messages(...)} only fills the templates; {@code llm.invoke(messages)} then sends them. Changing just
 * the variables - another domain, topic or audience - reuses the same prompt for a completely different explanation.
 *
 * <h2>Notebook result</h2>
 * A 3-line explanation of AWS Bedrock for senior architects: a fully managed service for building generative-AI
 * applications on foundation models, customisable without deep ML expertise, integrated with other AWS services.
 */
public final class MessagesTemplateMain {

    static final MessagesTemplate PROMPT = MessagesTemplate.fromMessages(
            "You are an expert in {domain}",
            "Explain {topic} to a {audience} in 3 lines");

    static final Map<String, String> VALUES =
            Map.of("domain", "Banking", "topic", "AWS BedRock", "audience", "Senior Architects");

    private MessagesTemplateMain() {
    }

    /**
     * Entry point.
     * <ol>
     *   <li>Switch the console to UTF-8 and check OPENAI_API_KEY ({@link Models#init()}).</li>
     *   <li>{@code prompt.format_messages(domain="Banking", topic="AWS BedRock", audience="Senior Architects")} -
     *       print the two filled messages so you can see exactly what is sent.</li>
     *   <li>{@code llm.invoke(format_msg)} with gpt-4o-mini and print {@code response.content}.</li>
     * </ol>
     *
     * @param args not used
     */
    public static void main(String[] args) {
        Models.init();
        Models.section("ChatPromptTemplate.from_messages(system, human) -> format_messages(...)");
        List<ChatMessage> messages = PROMPT.formatMessages(VALUES);
        messages.forEach(m -> System.out.println(m.type() + ": " + text(m)));

        Models.section("llm.invoke(format_msg).content");
        ChatModel llm = Models.openAi("gpt-4o-mini");
        System.out.println(llm.chat(messages).aiMessage().text());
    }

    private static String text(ChatMessage message) {
        return message instanceof dev.langchain4j.data.message.SystemMessage s ? s.text()
                : ((dev.langchain4j.data.message.UserMessage) message).singleText();
    }
}
