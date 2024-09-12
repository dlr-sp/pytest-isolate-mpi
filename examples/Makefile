.PHONY: all clean

EXAMPLES = $(wildcard test_*.py)
EXAMPLE_OUTPUTS = $(EXAMPLES:%=%.out)

all: $(EXAMPLE_OUTPUTS)

# Some examples intentionally fail, so ignore errors
%.py.out: %.py
	-pytest --verbose $< > $@

clean:
	find . -name '*.py.out' -exec rm -f {} +
