	͑?_F??@͑?_F??@!͑?_F??@	?ۋT?@?ۋT?@!?ۋT?@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$͑?_F??@?q??r?.@A"O???3?@Yqx??u?@*	aX9???@2F
Iterator::Modeld<J%<?@!??mM?qU@),???)<@1egLdfS@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate?t!VD@!?3Ϛ?+@)?v?ӂ'@1?ax=??+@:Preprocessing2U
Iterator::Model::ParallelMapV2?YJ???@!ў4X\ @)?YJ???@1ў4X\ @:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???????!7?? T???)Y??9??1?	??k??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice(???????!%?W????)(???????1%?W????:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?g%???@!i8???p,@)?hr1֑?1N?I?Т??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorr???_??!Z????)r???_??1Z????:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?^??H@!O?t?,@)?M?d?q?1??zb?ڈ?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 4.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?ۋT?@IBB???W@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?q??r?.@?q??r?.@!?q??r?.@      ??!       "      ??!       *      ??!       2	"O???3?@"O???3?@!"O???3?@:      ??!       B      ??!       J	qx??u?@qx??u?@!qx??u?@R      ??!       Z	qx??u?@qx??u?@!qx??u?@b      ??!       JCPU_ONLYY?ۋT?@b qBB???W@