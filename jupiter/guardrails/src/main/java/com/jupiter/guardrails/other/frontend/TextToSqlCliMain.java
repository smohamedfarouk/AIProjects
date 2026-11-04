package com.jupiter.guardrails.other.frontend;

import com.jupiter.guardrails.common.Tracing;
import com.jupiter.guardrails.other.backend.TextToSqlAgent;
import com.jupiter.guardrails.other.backend.TextToSqlAppMain;

import java.util.List;
import java.util.Scanner;

/**
 * Console replacement for the Streamlit UI in guardrails_frontend.py:
 * shows tables, guardrails and sample questions, then answers questions with guardrail badges.
 */
public class TextToSqlCliMain {

    static final List<String> SAMPLES = List.of(
            "How many students are enrolled in GenAI?",
            "Which student got the highest placement package?",
            "List all students from Bangalore",
            "What is the average package by course?",
            "Show total revenue from each course");

    public static void main(String[] args) {
        Tracing.init(TextToSqlAppMain.PROJECT);
        try (TextToSqlAgent agent = TextToSqlAppMain.newAgent()) {
            printHeader(agent);
            repl(agent);
        } catch (java.sql.SQLException e) {
            System.err.println("Database connection failed: " + e.getMessage());
            System.exit(1);
        } finally {
            Tracing.shutdown();
        }
    }

    private static void printHeader(TextToSqlAgent agent) {
        System.out.println("""
                ==================================================
                 Text-to-SQL Agent
                 Ask questions in plain English. Guardrails + LangSmith monitoring built in.
                ==================================================
                Connected to MySQL

                Active guardrails:
                 - Input screening (injection / destructive intent)
                 - PII detection & masking (in + out)
                 - Read-only SQL enforcement
                 - Sensitive-column blocking
                 - Hallucination check on answers

                Traces stream to LangSmith project `%s`.
                """.formatted(TextToSqlAppMain.PROJECT));
        System.out.println("Available tables: " + agent.getTableNames());
        System.out.println("\nSample questions:");
        for (int i = 0; i < SAMPLES.size(); i++) {
            System.out.println("  " + (i + 1) + ". " + SAMPLES.get(i));
        }
        System.out.println("\nType a question, a sample number, :schema to view the schema, or :quit to exit.");
    }

    private static void repl(TextToSqlAgent agent) {
        Scanner in = new Scanner(System.in);
        while (true) {
            System.out.print("\n> ");
            if (!in.hasNextLine()) {
                return;
            }
            String line = in.nextLine().strip();
            if (line.isEmpty()) {
                continue;
            }
            if (line.equals(":quit") || line.equals(":q")) {
                return;
            }
            if (line.equals(":schema")) {
                System.out.println(agent.getSchemaInfo());
                continue;
            }
            String question = line;
            if (line.matches("\\d+")) {
                int n = Integer.parseInt(line);
                if (n < 1 || n > SAMPLES.size()) {
                    System.out.println("No sample question " + n + ".");
                    continue;
                }
                question = SAMPLES.get(n - 1);
                System.out.println(question);
            }
            try {
                render(agent.query(question));
            } catch (RuntimeException e) {
                System.out.println("Error: " + e.getMessage());
            }
        }
    }

    private static void render(TextToSqlAgent.QueryResult r) {
        System.out.println("\n" + r.answer());
        System.out.println();
        System.out.println(r.blocked() ? "[BLOCKED] Input blocked" : "[OK] Input passed");
        System.out.println(r.piiFlags().isEmpty() ? "[OK] No PII" : "[WARN] PII masked: " + String.join(", ", r.piiFlags()));
        System.out.println(r.hallucinationFlag() ? "[WARN] Possible hallucination" : "[OK] Grounded");
        if (!r.sqlQuery().isEmpty()) {
            System.out.println("\nSQL:\n  " + r.sqlQuery());
        }
        if (!r.rawResult().isEmpty()) {
            System.out.println("\nRaw result:\n  " + r.rawResult());
        }
    }
}
