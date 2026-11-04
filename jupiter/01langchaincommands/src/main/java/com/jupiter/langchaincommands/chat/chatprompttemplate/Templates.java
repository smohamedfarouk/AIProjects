package com.jupiter.langchaincommands.chat.chatprompttemplate;

/**
 * The notebook's prompt templates and example input for the prompt lessons, in one place.
 * <p>
 * Used by {@link DraftEmailTemplateMain} and {@link MedicalTemplateMain} (each template as its own chain) and by the
 * reflection lesson
 * ({@code reflection.ReflectionChain}, which starts from {@link #DRAFT_TEMPLATE}) - so the single draft and the
 * reflected email are produced from exactly the same prompt and offer, and can be compared.
 * Placeholders are written as {@code {name}}, like LangChain's ChatPromptTemplate.
 */
public final class Templates {

    /** The notebook's draft_prompt - {@code {input}} is the client's offer. */
    public static final String DRAFT_TEMPLATE =
            "Draft a short polite email to a client declining their Corporate Training Proposal due to Low Budget: {input}";

    /** The notebook's medical_prompt - {@code {Name}} is the medicine. */
    public static final String MEDICAL_TEMPLATE = "Explain about this medicine Pros and Cons and side effects in 2Lines "
            + "and End with Thanks for Reaching us out www.bepec.in: {Name}";

    /** Example input for the draft and reflection lessons: the client's low-budget offer. */
    public static final String CLIENT_OFFER_INPUT_REASON =
            "The client offered 25000/day instead of 1lakh/day for AI Corporate Training";

    private Templates() {
    }
}
