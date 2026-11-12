package com.bepec.ragchunking;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/** Python-style {@code repr()} so the Java output can be compared line by line with the notebook's output. */
public final class Py {

    private Py() {
    }

    /** {@code repr(s)}: single quotes (double quotes if the text has a ' but no "), with \n, \t, \\ escaped. */
    public static String repr(String s) {
        char quote = s.indexOf('\'') >= 0 && s.indexOf('"') < 0 ? '"' : '\'';
        StringBuilder sb = new StringBuilder().append(quote);
        for (char c : s.toCharArray()) {
            switch (c) {
                case '\n' -> sb.append("\\n");
                case '\r' -> sb.append("\\r");
                case '\t' -> sb.append("\\t");
                case '\\' -> sb.append("\\\\");
                default -> {
                    if (c == quote) {
                        sb.append('\\').append(c);
                    } else if (c < 0x20 || c == 0x7f) {
                        sb.append(String.format("\\x%02x", (int) c));
                    } else {
                        sb.append(c);
                    }
                }
            }
        }
        return sb.append(quote).toString();
    }

    /** {@code repr(list_of_str)}, e.g. {@code ['a', 'b']}. */
    public static String repr(List<String> list) {
        return list.stream().map(Py::repr).collect(Collectors.joining(", ", "[", "]"));
    }

    /** {@code repr(dict_of_str)}, e.g. {@code {'H1': 'AI Basics'}}. */
    public static String repr(Map<String, String> map) {
        return map.entrySet().stream()
                .map(e -> repr(e.getKey()) + ": " + repr(e.getValue()))
                .collect(Collectors.joining(", ", "{", "}"));
    }
}
