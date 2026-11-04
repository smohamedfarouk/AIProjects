package com.bepec.deepagents.tools;

import dev.langchain4j.agent.tool.P;
import dev.langchain4j.agent.tool.Tool;

import java.util.List;

/** Python {@code expand_keywords(seed)}. */
public class ExpandKeywordsTool {

    static final List<String> MODIFIERS = List.of("course", "training", "online", "certification",
            "for beginners", "with placements", "professionals", "2026");

    @Tool(name = "expand_keywords", value = """
            Expand a seed keyword into long-tail variations using common modifiers. \
            use this when you need keyword ideas for seo or google ads""")
    public List<String> expandKeywords(@P("the seed keyword") String seed) {
        return MODIFIERS.stream().map(m -> seed + " " + m).toList();
    }
}
