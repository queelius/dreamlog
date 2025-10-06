# Makefile for building DreamLog research paper
# Author: DreamLog Team

# Main target
PAPER = dreamlog_paper
TEX = $(PAPER).tex
PDF = $(PAPER).pdf
BIB = references.bib

# LaTeX compiler
LATEX = pdflatex
BIBTEX = bibtex

# Flags
LATEX_FLAGS = -interaction=nonstopmode -file-line-error

# Default target
all: $(PDF)

# Build the PDF (full compilation with bibliography)
$(PDF): $(TEX)
	@echo "Building $(PDF)..."
	$(LATEX) $(LATEX_FLAGS) $(TEX)
	@if [ -f $(BIB) ]; then \
		$(BIBTEX) $(PAPER); \
		$(LATEX) $(LATEX_FLAGS) $(TEX); \
	fi
	$(LATEX) $(LATEX_FLAGS) $(TEX)
	@echo "Build complete: $(PDF)"

# Quick build (single pass, no bibliography)
quick: $(TEX)
	@echo "Quick build of $(PDF)..."
	$(LATEX) $(LATEX_FLAGS) $(TEX)
	@echo "Quick build complete: $(PDF)"

# Clean auxiliary files
clean:
	@echo "Cleaning auxiliary files..."
	rm -f *.aux *.log *.out *.toc *.lof *.lot *.bbl *.blg
	rm -f *.nav *.snm *.vrb *.fls *.fdb_latexmk *.synctex.gz
	@echo "Clean complete"

# Clean everything including PDF
distclean: clean
	@echo "Removing $(PDF)..."
	rm -f $(PDF)
	@echo "Distclean complete"

# View the PDF
view: $(PDF)
	@if command -v xdg-open > /dev/null; then \
		xdg-open $(PDF); \
	elif command -v open > /dev/null; then \
		open $(PDF); \
	else \
		echo "Please open $(PDF) manually"; \
	fi

# Check for LaTeX installation
check:
	@echo "Checking LaTeX installation..."
	@which $(LATEX) > /dev/null 2>&1 || (echo "Error: pdflatex not found. Please install LaTeX." && exit 1)
	@which $(BIBTEX) > /dev/null 2>&1 || (echo "Warning: bibtex not found. Bibliography processing will be skipped.")
	@echo "LaTeX installation check complete"

# Count words in the paper (approximate)
wordcount:
	@echo "Approximate word count:"
	@detex $(TEX) 2>/dev/null | wc -w || echo "detex not found, using basic count: $$(grep -v '^%' $(TEX) | wc -w) words"

# Help
help:
	@echo "DreamLog Paper Makefile"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all       - Build the complete PDF with bibliography (default)"
	@echo "  quick     - Quick build (single LaTeX pass, no bibliography)"
	@echo "  clean     - Remove auxiliary files"
	@echo "  distclean - Remove all generated files including PDF"
	@echo "  view      - Open the PDF in default viewer"
	@echo "  check     - Check for LaTeX installation"
	@echo "  wordcount - Count approximate words in the paper"
	@echo "  help      - Show this help message"
	@echo ""
	@echo "Example:"
	@echo "  make          # Build the paper"
	@echo "  make clean    # Clean up auxiliary files"
	@echo "  make view     # View the PDF"

.PHONY: all quick clean distclean view check wordcount help