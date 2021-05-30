all: report.pdf

report.pdf: report.tex
	pdflatex report.tex
	bibtex report
	pdflatex report.tex
	pdflatex report.tex

.PHONY: open clean

open: report.pdf
	gnome-open report.pdf

clean:
	echo "todo"
