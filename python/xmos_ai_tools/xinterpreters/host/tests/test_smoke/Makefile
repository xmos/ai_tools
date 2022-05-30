# TODO: do all in python, use pytest
# TODO: derive smoke_model.flash from smoke_model.params
# TODO: derive smoke_model.params and smoke_model.tflite from tflite
test:
	python3 test_smoke.py
	@diff out0 classes_gt.raw
	@diff out1 boxes_gt.raw
	@rm -f out0 out1
	@echo 'PASS: test_smoke'
