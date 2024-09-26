cp $1 /tmp/
xcore-opt /tmp/$1 --lce-translate-tfl --mlir-print-ir-after-all -o /tmp/1.tflite >/tmp/1.mlir 2>&1
cat /tmp/1.mlir | grep -v Tensor > /tmp/2.mlir
sed -i -e 's/tfl.add/tfl.sub/g' /tmp/2.mlir
xcore-opt --mlir-io --lce-translate-tfl /tmp/2.mlir -o /tmp/t.tflite
cp /tmp/t.tflite $1
