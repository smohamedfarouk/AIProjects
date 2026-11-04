package com.jupiter.guardrails.common.eval;

import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.function.Function;

/**
 * Java equivalent of langsmith.evaluation.evaluate(): runs the target on every dataset example,
 * logs each run to a new experiment and attaches every evaluator's score as feedback.
 */
public final class ExperimentRunner {

    /** One evaluated example. */
    public record Row(Map<String, Object> inputs, Map<String, Object> outputs, String error,
                      List<Evaluator.Result> results) {
    }

    private ExperimentRunner() {
    }

    public static List<Row> evaluate(LangSmithClient ls,
                                     Function<Map<String, Object>, Map<String, Object>> target,
                                     String datasetName,
                                     List<Evaluator> evaluators,
                                     String experimentPrefix) {
        String datasetId = ls.findDatasetId(datasetName).orElseThrow(() ->
                new IllegalStateException("Dataset '" + datasetName + "' not found in LangSmith. Create it first."));
        List<LangSmithClient.Example> examples = ls.listExamples(datasetId);

        String experimentName = experimentPrefix + "-" + UUID.randomUUID().toString().substring(0, 8);
        String experimentId = ls.createExperiment(experimentName, datasetId);
        System.out.println("Experiment '" + experimentName + "' on dataset '" + datasetName
                + "' (" + examples.size() + " examples)\n");

        List<Row> rows = new ArrayList<>();
        for (LangSmithClient.Example example : examples) {
            UUID runId = UUID.randomUUID();
            Instant start = Instant.now();
            Map<String, Object> outputs = Map.of();
            String error = null;
            try {
                outputs = target.apply(example.inputs());
            } catch (RuntimeException e) {
                error = e.toString();
            }
            ls.createRun(runId, "target", experimentId, example.id(),
                    example.inputs(), outputs, error, start, Instant.now());

            Evaluator.Case c = new Evaluator.Case(example.inputs(), outputs, example.outputs());
            List<Evaluator.Result> results = new ArrayList<>();
            for (Evaluator evaluator : evaluators) {
                try {
                    Evaluator.Result r = evaluator.evaluate(c);
                    ls.createFeedback(runId, r.key(), r.score(), r.comment());
                    results.add(r);
                } catch (RuntimeException e) {
                    System.err.println("Evaluator failed on " + example.inputs() + ": " + e);
                }
            }
            rows.add(new Row(example.inputs(), outputs, error, results));
            printRow(rows.size(), rows.get(rows.size() - 1));
        }
        ls.endExperiment(experimentId);
        printSummary(rows);
        return rows;
    }

    private static void printRow(int n, Row row) {
        System.out.println(n + ". " + row.inputs());
        System.out.println("   output: " + (row.error() != null ? "ERROR " + row.error() : row.outputs()));
        for (Evaluator.Result r : row.results()) {
            System.out.println("   " + r.key() + " = " + r.score() + (r.comment() != null ? "  (" + r.comment() + ")" : ""));
        }
        System.out.println();
    }

    private static void printSummary(List<Row> rows) {
        Map<String, double[]> totals = new LinkedHashMap<>(); // key -> {sum, count}
        for (Row row : rows) {
            for (Evaluator.Result r : row.results()) {
                double[] t = totals.computeIfAbsent(r.key(), k -> new double[2]);
                if (r.score() != null) {
                    t[0] += r.score();
                    t[1]++;
                }
            }
        }
        System.out.println("===== Average scores =====");
        totals.forEach((key, t) -> System.out.printf("%-20s %s%n", key,
                t[1] == 0 ? "n/a" : String.format("%.2f (%d scored)", t[0] / t[1], (int) t[1])));
    }
}
