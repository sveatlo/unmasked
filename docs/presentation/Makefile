all: prezentace.pdf

.PHONY: clean open

open: prezentace.pdf
	gnome-open prezentace.pdf

clean:
	rm prezentace.pdf

%.pdf: %.tex
	pdflatex $<
	pdflatex $<
