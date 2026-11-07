package com.bepec.ragpinecone;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.Metadata;
import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.text.PDFTextStripper;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Java version of {@code PyPDFLoader(PDF_PATH).load()}: one {@link Document} per PDF page.
 */
public final class PdfLoader {

    private PdfLoader() {
    }

    /**
     * Reads a PDF and returns one Document per page, like PyPDFLoader.
     * <ol>
     *   <li>Open the PDF with Apache PDFBox.</li>
     *   <li>For each page, extract only that page's text ({@link PDFTextStripper} with start page = end page).</li>
     *   <li>Attach metadata the rest of the pipeline relies on: {@code page} (0-based, exactly like PyPDFLoader, so
     *       page 1 of the PDF has page = 0), {@code source} (file name) and {@code total_pages}.</li>
     * </ol>
     * Pages with no extractable text (e.g. a pure image) are kept as empty documents so page numbers stay aligned;
     * the splitter later produces no chunks for them. Scanned PDFs without a text layer would need OCR.
     *
     * @param pdf path to the PDF file
     * @return one document per page, in page order
     * @throws IOException if the file is missing or is not a readable PDF
     */
    public static List<Document> load(Path pdf) throws IOException {
        if (!Files.isRegularFile(pdf)) {
            throw new IOException("PDF not found: " + pdf.toAbsolutePath());
        }
        List<Document> pages = new ArrayList<>();
        try (PDDocument document = Loader.loadPDF(pdf.toFile())) {
            int total = document.getNumberOfPages();
            PDFTextStripper stripper = new PDFTextStripper();
            for (int page = 1; page <= total; page++) {
                stripper.setStartPage(page);
                stripper.setEndPage(page);
                String text = stripper.getText(document);

                Metadata metadata = new Metadata()
                        .put("source", pdf.getFileName().toString())
                        .put("page", page - 1)
                        .put("total_pages", total);
                pages.add(Document.from(text == null ? "" : text, metadata));
            }
        }
        return pages;
    }
}
