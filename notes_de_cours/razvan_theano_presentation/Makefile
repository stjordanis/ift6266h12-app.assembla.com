FILES = presentation.aux presentation.blg presentation.dvi presentation.log presentation.ps \
		presentation.pdf presentation.toc presentation.bbl presentation.nav presentation.out \
		presentation.snm presentation.vrb

all: presentation.tex
	pdflatex --shell-escape presentation.tex
	pdflatex --shell-escape presentation.tex

clean:
	rm -f $(FILES)
