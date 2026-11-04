package com.jupiter.langchaincommands.chat.chatprompttemplate;

import dev.langchain4j.model.chat.ChatModel;

import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Java version of the LCEL chain {@code ChatPromptTemplate.from_template(t) | llm | StrOutputParser()}:
 * fills the {name} placeholders, sends the result as a user message and returns the reply text.
 * <pre>
 *   template "Explain about this medicine ...: {Name}"  +  {"Name": "Creatine"}
 *     -> prompt "Explain about this medicine ...: Creatine"  -> llm  -> reply text
 * </pre>
 * One template can be reused for any value - that is the point of prompt templates.
 */
public final class PromptChain {

    private static final Pattern VARIABLE = Pattern.compile("\\{(\\w+)}");

    private final String template;
    private final ChatModel llm;

    private PromptChain(String template, ChatModel llm) {
        this.template = template;
        this.llm = llm;
    }

    /**
     * Creates a chain - {@code ChatPromptTemplate.from_template(template) | llm | StrOutputParser()}.
     *
     * @param template prompt text with {@code {name}} placeholders (letters, digits, underscore)
     * @param llm      the model that answers the filled-in prompt
     */
    public static PromptChain fromTemplate(String template, ChatModel llm) {
        return new PromptChain(template, llm);
    }

    /**
     * Runs the chain - {@code chain.invoke({"Name": "Creatine"})}: fill the placeholders ({@link #format}), send the
     * prompt to the model as one user message, return the reply as plain text.
     *
     * @param variables a value for every placeholder in the template (extra entries are ignored)
     * @return the model's reply text
     * @throws IllegalArgumentException if a placeholder has no value
     */
    public String invoke(Map<String, ?> variables) {
        return llm.chat(format(variables));
    }

    /**
     * Shortcut like LangChain's {@code chain.invoke("text")}: a plain string fills the template's first (normally
     * only) placeholder. Used by the reflection chain's draft step.
     */
    public String invoke(String value) {
        Matcher m = VARIABLE.matcher(template);
        if (!m.find()) {
            throw new IllegalArgumentException("Template has no variables: " + template);
        }
        return invoke(Map.of(m.group(1), value));
    }

    /**
     * Fills every {@code {name}} placeholder with its value - the "prompt" step on its own, without calling the
     * model. Values are inserted literally, so text containing $ or \ is safe.
     */
    String format(Map<String, ?> variables) {
        return fill(template, variables);
    }

    /**
     * Fills the {@code {name}} placeholders of any template - shared with {@link MessagesTemplate}, which fills a
     * system and a human template the same way.
     *
     * @throws IllegalArgumentException if a placeholder has no value
     */
    static String fill(String template, Map<String, ?> variables) {
        Matcher m = VARIABLE.matcher(template);
        StringBuilder out = new StringBuilder();
        while (m.find()) {
            Object value = variables.get(m.group(1));
            if (value == null) {
                throw new IllegalArgumentException("Missing variable '" + m.group(1) + "' for prompt template");
            }
            m.appendReplacement(out, Matcher.quoteReplacement(value.toString()));
        }
        m.appendTail(out);
        return out.toString();
    }
}
