	`??i?Y@`??i?Y@!`??i?Y@	ZG??,/@ZG??,/@!ZG??,/@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$`??i?Y@?~?Ϛ??A3??(KX@Y???7??@*	??v????@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?^}<?m@!?]??_?Q@)-$`tyS@1????Q@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip&Q/?4?@!??r??X@)?>???
??1N?s?:(5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateM???$??!h??^?%@)[?:?????1????\a@:Preprocessing2U
Iterator::Model::ParallelMapV2?2o?u???!??b
????)?2o?u???1??b
????:Preprocessing2F
Iterator::ModelE.8??_??!^M????)?)1	??1?7<????:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???sE)??!_rV$??)???sE)??1_rV$??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorj?:?z??!|??
YM??)j?:?z??1|??
YM??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?)? ?h??!?|?j{@)&R???0x?1?Z<??I??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 5.3% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no9ZG??,/@I???5?W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?~?Ϛ???~?Ϛ??!?~?Ϛ??      ??!       "      ??!       *      ??!       2	3??(KX@3??(KX@!3??(KX@:      ??!       B      ??!       J	???7??@???7??@!???7??@R      ??!       Z	???7??@???7??@!???7??@b      ??!       JCPU_ONLYYZG??,/@b q???5?W@