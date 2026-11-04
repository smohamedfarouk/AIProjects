package com.jupiter.guardrails.other.backend;

import com.jupiter.guardrails.common.Tracing;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Rule-based guardrails for the text-to-SQL agent: input screening, PII detection/masking, read-only SQL checks. */
public final class SqlGuardrails {

    public static final List<String> DESTRUCTIVE_KEYWORDS = List.of(
            "DROP", "DELETE", "UPDATE", "TRUNCATE", "ALTER",
            "INSERT", "CREATE", "GRANT", "REVOKE", "REPLACE", "MERGE");

    static final List<Pattern> INJECTION_PATTERNS = compileAll(
            "ignore (all |the )?(previous|prior|above) instructions",
            "disregard .* (instructions|rules|prompt)",
            "system prompt",
            "you are now",
            "reveal .* (prompt|schema secrets|credentials|password)",
            "act as (?!a mysql)",  // allow "act as a MySQL expert", block other role-plays
            "drop\\s+table",
            "delete\\s+from",
            ";\\s*(drop|delete|update|insert|alter|truncate)");

    static final Map<String, Pattern> PII_PATTERNS = new LinkedHashMap<>();

    static {
        PII_PATTERNS.put("EMAIL", Pattern.compile("\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}\\b"));
        PII_PATTERNS.put("PHONE", Pattern.compile("\\b(?:\\+?\\d{1,3}[\\s-]?)?(?:\\d{10}|\\d{5}[\\s-]?\\d{5})\\b"));
        PII_PATTERNS.put("AADHAAR", Pattern.compile("\\b\\d{4}\\s?\\d{4}\\s?\\d{4}\\b"));
        PII_PATTERNS.put("PAN", Pattern.compile("\\b[A-Z]{5}\\d{4}[A-Z]\\b"));
        PII_PATTERNS.put("CREDIT_CARD", Pattern.compile("\\b(?:\\d[ -]*?){13,16}\\b"));
        PII_PATTERNS.put("SSN", Pattern.compile("\\b\\d{3}-\\d{2}-\\d{4}\\b"));
        PII_PATTERNS.put("IP", Pattern.compile("\\b(?:\\d{1,3}\\.){3}\\d{1,3}\\b"));
    }

    static final Set<String> SENSITIVE_COLUMNS = Set.of(
            "password", "passwd", "pwd", "email", "phone", "mobile", "contact",
            "aadhaar", "aadhar", "pan", "ssn", "credit_card", "card_number",
            "cvv", "dob", "date_of_birth", "address", "salary_account");

    static final int MAX_QUESTION_LENGTH = 1000;

    /** Outcome of the input guardrail; reason is empty when allowed. */
    public record InputVerdict(boolean allowed, String reason) {
    }

    private SqlGuardrails() {
    }

    /** Return list of PII labels found in text. */
    public static List<String> detectPii(String text) {
        return Tracing.trace("detect_pii", "chain", span -> {
            List<String> found = new ArrayList<>();
            if (text == null || text.isEmpty()) {
                return found;
            }
            PII_PATTERNS.forEach((label, pattern) -> {
                if (pattern.matcher(text).find()) {
                    found.add(label);
                }
            });
            span.setAttribute("output.value", found.toString());
            return found;
        });
    }

    /** Replace detected PII with typed placeholders. */
    public static String maskPii(String text) {
        return Tracing.trace("mask_pii", "chain", span -> {
            if (text == null || text.isEmpty()) {
                return text;
            }
            String masked = text;
            for (Map.Entry<String, Pattern> e : PII_PATTERNS.entrySet()) {
                masked = e.getValue().matcher(masked).replaceAll(Matcher.quoteReplacement("[REDACTED_" + e.getKey() + "]"));
            }
            return masked;
        });
    }

    /** Screen user input for injection, destructive intent, empty/oversized input. */
    public static InputVerdict checkInputGuardrail(String question) {
        return Tracing.trace("input_guardrail", "chain", span -> {
            InputVerdict verdict = screen(question == null ? "" : question.strip());
            Tracing.metadata(span, "guard.allowed", verdict.allowed());
            Tracing.metadata(span, "guard.reason", verdict.reason());
            return verdict;
        });
    }

    private static InputVerdict screen(String q) {
        if (q.isEmpty()) {
            return new InputVerdict(false, "Empty question.");
        }
        if (q.length() > MAX_QUESTION_LENGTH) {
            return new InputVerdict(false, "Question too long (max 1000 chars).");
        }
        String lowered = q.toLowerCase(Locale.ROOT);
        for (Pattern p : INJECTION_PATTERNS) {
            if (p.matcher(lowered).find()) {
                return new InputVerdict(false, "Input blocked: possible prompt-injection or unsafe instruction.");
            }
        }
        String kw = findDestructiveKeyword(q);
        if (kw != null) {
            return new InputVerdict(false, "Input blocked: destructive keyword '" + kw + "' detected. Read-only mode.");
        }
        return new InputVerdict(true, "");
    }

    /** First destructive SQL keyword appearing as a whole word (case-insensitive), or null. */
    public static String findDestructiveKeyword(String text) {
        List<String> all = findDestructiveKeywords(text);
        return all.isEmpty() ? null : all.get(0);
    }

    public static List<String> findDestructiveKeywords(String text) {
        String upper = text.toUpperCase(Locale.ROOT);
        return DESTRUCTIVE_KEYWORDS.stream()
                .filter(kw -> Pattern.compile("\\b" + kw + "\\b").matcher(upper).find())
                .toList();
    }

    /** First sensitive column name referenced in the SQL, or null. */
    public static String findSensitiveColumn(String sql) {
        String lower = sql.toLowerCase(Locale.ROOT);
        for (String col : SENSITIVE_COLUMNS) {
            if (Pattern.compile("\\b" + col + "\\b").matcher(lower).find()) {
                return col;
            }
        }
        return null;
    }

    private static List<Pattern> compileAll(String... regexes) {
        return java.util.Arrays.stream(regexes).map(Pattern::compile).toList();
    }
}
