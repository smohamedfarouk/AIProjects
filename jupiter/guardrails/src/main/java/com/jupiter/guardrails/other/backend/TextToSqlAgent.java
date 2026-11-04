package com.jupiter.guardrails.other.backend;

import com.jupiter.guardrails.common.Config;
import com.jupiter.guardrails.common.LlmClient;
import com.jupiter.guardrails.common.Tracing;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Java port of TextToSQLAgent from guardrails_backend.py. The LangGraph pipeline runs as plain method calls:
 * input_guardrail -> get_schema -> generate_sql -> validate_sql -> execute_sql -> generate_answer,
 * skipping straight to generate_answer when a guardrail sets an error.
 */
public class TextToSqlAgent implements AutoCloseable {

    static final int SAMPLE_ROWS = 3;
    static final int MAX_VALUE_LENGTH = 100;

    static final Pattern SCHEMA_COLUMN = Pattern.compile("\\s*`?(\\w+)`?\\s+\\w+");
    static final Pattern NUMBER = Pattern.compile("\\d+(?:\\.\\d+)?");

    static final String SQL_PROMPT = """

            You are a MySQL expert. Based ONLY on the schema below, write a syntactically correct MySQL query to answer the user's question.

            Schema:
            %s

            Question: %s

            Rules:
            - Use ONLY tables and columns that appear in the schema above. Never invent names.
            - Do NOT select password, email, phone, aadhaar, pan, ssn, card, cvv, dob, or address columns.
            - Return ONLY the SQL query, no explanations, no markdown, no backticks.
            - Use proper JOINs when needed.
            - Use LIMIT 10 unless the user specifies otherwise.
            - Never use DROP, DELETE, UPDATE, TRUNCATE, ALTER, INSERT, CREATE, GRANT, REVOKE.

            SQL Query:""";

    static final String ANSWER_PROMPT = """

            Answer the user's question using ONLY the SQL result provided. Do not add facts,
            numbers, or names that are not present in the result. If the result is empty or an
            error, say so plainly. Do not speculate.

            Question: %s
            SQL Query: %s
            Result: %s

            Answer:""";

    /** Final output of one query (same fields as the Python query() dict). */
    public record QueryResult(String question, String sqlQuery, String rawResult, String answer,
                              List<String> piiFlags, boolean hallucinationFlag, boolean blocked) {

        public Map<String, Object> toMap() {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("question", question);
            m.put("sql_query", sqlQuery);
            m.put("raw_result", rawResult);
            m.put("answer", answer);
            m.put("pii_flags", piiFlags);
            m.put("hallucination_flag", hallucinationFlag);
            m.put("blocked", blocked);
            return m;
        }
    }

    /** Mirrors the Python AgentState TypedDict. */
    static class AgentState {
        String question;
        String schema = "";
        String sqlQuery = "";
        String queryResult = "";
        String finalAnswer = "";
        String error = "";
        List<String> piiFlags = List.of();
        boolean hallucinationFlag = false;

        AgentState(String question) {
            this.question = question;
        }

        boolean hasError() {
            return error != null && !error.isEmpty();
        }
    }

    private final Connection db;
    private final LlmClient llm;
    private final List<String> schemaColumns;

    public TextToSqlAgent(LlmClient llm) throws SQLException {
        this.db = connectDb();
        this.llm = llm;
        this.schemaColumns = extractSchemaColumns();
    }

    static final List<String> REQUIRED_SETTINGS =
            List.of("MYSQL_HOST", "MYSQL_PORT", "MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_DATABASE");

    private static Connection connectDb() throws SQLException {
        List<String> missing = REQUIRED_SETTINGS.stream().filter(k -> Config.get(k) == null).toList();
        if (!missing.isEmpty()) {
            throw new SQLException("Missing MySQL settings " + missing + ". Add them to jupiter/.env or the environment.");
        }
        String url = "jdbc:mysql://" + Config.get("MYSQL_HOST") + ":" + Config.get("MYSQL_PORT")
                + "/" + Config.get("MYSQL_DATABASE");
        return DriverManager.getConnection(url, Config.get("MYSQL_USER"), Config.get("MYSQL_PASSWORD"));
    }

    public List<String> getTableNames() {
        List<String> tables = new ArrayList<>();
        try (Statement st = db.createStatement(); ResultSet rs = st.executeQuery("SHOW TABLES")) {
            while (rs.next()) {
                tables.add(rs.getString(1));
            }
        } catch (SQLException e) {
            throw new IllegalStateException("Could not list tables", e);
        }
        tables.sort(String::compareTo);
        return tables;
    }

    /** CREATE TABLE statement plus a few sample rows per table (like SQLDatabase.get_table_info). */
    public String getSchemaInfo() {
        List<String> parts = new ArrayList<>();
        for (String table : getTableNames()) {
            StringBuilder sb = new StringBuilder();
            try (Statement st = db.createStatement()) {
                try (ResultSet rs = st.executeQuery("SHOW CREATE TABLE `" + table + "`")) {
                    if (rs.next()) {
                        sb.append('\n').append(rs.getString(2)).append("\n\n");
                    }
                }
                try (ResultSet rs = st.executeQuery("SELECT * FROM `" + table + "` LIMIT " + SAMPLE_ROWS)) {
                    ResultSetMetaData md = rs.getMetaData();
                    sb.append("/*\n").append(SAMPLE_ROWS).append(" rows from ").append(table).append(" table:\n");
                    List<String> header = new ArrayList<>();
                    for (int i = 1; i <= md.getColumnCount(); i++) {
                        header.add(md.getColumnLabel(i));
                    }
                    sb.append(String.join("\t", header)).append('\n');
                    while (rs.next()) {
                        List<String> row = new ArrayList<>();
                        for (int i = 1; i <= md.getColumnCount(); i++) {
                            row.add(truncate(String.valueOf(rs.getObject(i))));
                        }
                        sb.append(String.join("\t", row)).append('\n');
                    }
                    sb.append("*/");
                }
            } catch (SQLException e) {
                throw new IllegalStateException("Could not read schema for " + table, e);
            }
            parts.add(sb.toString());
        }
        return String.join("\n\n", parts);
    }

    /** Pull column + table names from schema for hallucination grounding. */
    private List<String> extractSchemaColumns() {
        Set<String> tokens = new HashSet<>();
        for (String table : getTableNames()) {
            tokens.add(table.toLowerCase(Locale.ROOT));
        }
        // crude column capture from CREATE TABLE lines
        for (String line : getSchemaInfo().split("\\R")) {
            Matcher m = SCHEMA_COLUMN.matcher(line);
            if (m.lookingAt()) {
                tokens.add(m.group(1).toLowerCase(Locale.ROOT));
            }
        }
        return new ArrayList<>(tokens);
    }

    public List<String> schemaColumns() {
        return schemaColumns;
    }

    // ---------- graph nodes ----------

    void inputGuardrail(AgentState state) {
        Tracing.trace("node_input_guardrail", "chain", span -> {
            List<String> pii = SqlGuardrails.detectPii(state.question);
            SqlGuardrails.InputVerdict verdict = SqlGuardrails.checkInputGuardrail(state.question);
            state.error = verdict.allowed() ? "" : verdict.reason();
            state.piiFlags = pii;
        });
    }

    void getSchema(AgentState state) {
        state.schema = getSchemaInfo();
    }

    void generateSql(AgentState state) {
        Tracing.trace("node_generate_sql", "chain", span -> {
            String response = llm.ask(SQL_PROMPT.formatted(state.schema, state.question));
            state.sqlQuery = response.strip().replace("```sql", "").replace("```", "").strip();
            span.setAttribute("output.value", state.sqlQuery);
        });
    }

    /** Destructive-keyword block + sensitive-column block + schema grounding. */
    void validateSql(AgentState state) {
        Tracing.trace("node_validate_sql", "chain", span -> {
            String sql = state.sqlQuery;
            String kw = SqlGuardrails.findDestructiveKeyword(sql);
            String col = SqlGuardrails.findSensitiveColumn(sql);
            long semicolons = sql.chars().filter(ch -> ch == ';').count();

            if (kw != null) {
                state.error = "Blocked: query contains '" + kw + "'. Read-only mode.";
            } else if (semicolons > 1 || (semicolons == 1 && !sql.strip().endsWith(";"))) {
                state.error = "Blocked: multiple SQL statements are not allowed.";
            } else if (col != null) {
                state.error = "Blocked: query references sensitive column '" + col + "'.";
            } else {
                state.error = "";
            }
            Tracing.metadata(span, "guard.blocked", state.hasError());
            Tracing.metadata(span, "guard.reason", state.error);
        });
    }

    void executeSql(AgentState state) {
        Tracing.trace("node_execute_sql", "chain", span -> {
            try {
                String result = runSql(state.sqlQuery);
                String raw = result.isEmpty() ? "No results found." : result;
                List<String> leaked = SqlGuardrails.detectPii(raw);
                if (!leaked.isEmpty()) {
                    raw = SqlGuardrails.maskPii(raw);
                }
                state.queryResult = raw;
                state.piiFlags = leaked;
            } catch (SQLException e) {
                state.queryResult = "Error executing query: " + e.getMessage();
            }
        });
    }

    void generateAnswer(AgentState state) {
        Tracing.trace("node_generate_answer", "chain", span -> {
            if (state.hasError()) {
                state.finalAnswer = state.error;
                state.hallucinationFlag = false;
                return;
            }
            String answer = llm.ask(ANSWER_PROMPT.formatted(state.question, state.sqlQuery, state.queryResult));
            if (!SqlGuardrails.detectPii(answer).isEmpty()) {
                answer = SqlGuardrails.maskPii(answer);
            }
            state.finalAnswer = answer;
            state.hallucinationFlag = detectHallucination(answer, state.queryResult);
        });
    }

    /** Heuristic: any number in the answer must appear in the raw result. */
    boolean detectHallucination(String answer, String result) {
        return Tracing.trace("detect_hallucination", "chain", span -> {
            if (result == null || result.isEmpty() || result.contains("Error") || result.contains("No results")) {
                return false;
            }
            Set<String> ansNums = numbers(answer);
            Set<String> resNums = numbers(result);
            Set<Double> resValues = new HashSet<>();
            for (String n : resNums) {
                resValues.add(Double.parseDouble(n));
            }
            for (String n : ansNums) {
                if (!resNums.contains(n) && !resValues.contains(Double.parseDouble(n))) {
                    return true;
                }
            }
            return false;
        });
    }

    public QueryResult query(String question) {
        return Tracing.trace("text_to_sql_query", "chain", span -> {
            span.setAttribute("input.value", question);
            AgentState state = new AgentState(question);

            inputGuardrail(state);
            if (!state.hasError()) {
                getSchema(state);
                generateSql(state);
                validateSql(state);
                if (!state.hasError()) {
                    executeSql(state);
                }
            }
            generateAnswer(state);

            QueryResult result = new QueryResult(question, state.sqlQuery, state.queryResult, state.finalAnswer,
                    state.piiFlags, state.hallucinationFlag, state.hasError());
            span.setAttribute("output.value", result.answer());
            Tracing.metadata(span, "blocked", result.blocked());
            Tracing.metadata(span, "pii_flags", result.piiFlags());
            Tracing.metadata(span, "hallucination_flag", result.hallucinationFlag());
            return result;
        });
    }

    /** Runs the query and formats rows like Python's str(list_of_tuples), e.g. [(1, 'Alice'), (2, 'Bob')]. */
    String runSql(String sql) throws SQLException {
        try (Statement st = db.createStatement()) {
            if (!st.execute(sql)) {
                return "";
            }
            try (ResultSet rs = st.getResultSet()) {
                int cols = rs.getMetaData().getColumnCount();
                List<String> rows = new ArrayList<>();
                while (rs.next()) {
                    List<String> values = new ArrayList<>();
                    for (int i = 1; i <= cols; i++) {
                        values.add(pyRepr(rs.getObject(i)));
                    }
                    rows.add("(" + String.join(", ", values) + (cols == 1 ? ",)" : ")"));
                }
                return rows.isEmpty() ? "" : "[" + String.join(", ", rows) + "]";
            }
        }
    }

    private static String pyRepr(Object v) {
        if (v == null) return "None";
        if (v instanceof Boolean b) return b ? "True" : "False";
        if (v instanceof Number) return v.toString();
        return "'" + truncate(v.toString()).replace("\\", "\\\\").replace("'", "\\'") + "'";
    }

    private static String truncate(String s) {
        return s.length() <= MAX_VALUE_LENGTH ? s : s.substring(0, MAX_VALUE_LENGTH) + "...";
    }

    private static Set<String> numbers(String text) {
        Set<String> found = new HashSet<>();
        Matcher m = NUMBER.matcher(text);
        while (m.find()) {
            found.add(m.group());
        }
        return found;
    }

    @Override
    public void close() throws SQLException {
        db.close();
    }
}
