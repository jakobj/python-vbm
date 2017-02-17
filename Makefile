all: vbm.so

PHONY: clean

vbm.so: vbm.pyx vbm.h setup.py
	python setup.py build_ext -i

clean:
	rm -rf vbm.c vbm.so build/
