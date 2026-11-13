# One command per deliverable. Details: docs/technical_document.md, section 7.
# Override the interpreter with: make submission PYTHON=.venv/bin/python

PYTHON ?= python

.PHONY: submission rules selftest

# Dog_Lovers.csv: weights, game file, ball, possession, decoder twice, validator (CPU)
submission:
	$(PYTHON) -m pipeline.run

# Fallback without trained models: Dog_Lovers_rules.csv
rules:
	$(PYTHON) -m pipeline.run --rules

# Synthetic check of the whole chain, no match data
selftest:
	$(PYTHON) -m pipeline.run --selftest
