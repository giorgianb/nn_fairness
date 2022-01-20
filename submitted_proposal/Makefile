texfile='main'

all: build

build:
	pdflatex -synctex=1 --shell-escape -enable-write18 $(texfile).tex
	bibtex $(texfile).aux
	pdflatex -synctex=1 --shell-escape -enable-write18 $(texfile).tex
	pdflatex -synctex=1 --shell-escape -enable-write18 $(texfile).tex

.PHONY: clean
clean:
	rm $(texfile).aux
	rm $(texfile).dvi
	rm $(texfile).log
	rm $(texfile).nav
	rm $(texfile).toc
	rm $(texfile).out
	rm $(texfile).pdf
	rm $(texfile).snm
