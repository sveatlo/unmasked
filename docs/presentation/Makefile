all: presentation.pdf

.PHONY: clean open

open: presentation.pdf
	gnome-open presentation.pdf

clean:
	rm presentation.pdf

%.pdf: %.tex
	pdflatex $<
	pdflatex $<
