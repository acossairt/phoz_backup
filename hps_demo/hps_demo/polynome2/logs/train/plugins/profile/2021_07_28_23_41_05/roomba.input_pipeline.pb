	?YI+???@?YI+???@!?YI+???@	K?D?8@K?D?8@!K?D?8@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?YI+???@b?qm???AM????E?@Y?G?`??D@*	? ?r??@2U
Iterator::Model::ParallelMapV2^/M??D@!Y?P3?;L@)^/M??D@1Y?P3?;L@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?L????@!?y?ϨE@)??zi???@1???7?E@:Preprocessing2F
Iterator::Models????D@!V??FL@)?T??7??1????????:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?)Wx????!?u?f??)1%??e??1??b"~T??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSliceo~?D???!??{????)o~?D???1??{????:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?s??@@!??k?S?E@)l#???1	d?t????:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorиp $??!?[Hp_F??)иp $??1?[Hp_F??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?Z??v @@!?~B??E@)??nI?u?1??'?r?}?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 6.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no9L?D?8@I???ku|W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	b?qm???b?qm???!b?qm???      ??!       "      ??!       *      ??!       2	M????E?@M????E?@!M????E?@:      ??!       B      ??!       J	?G?`??D@?G?`??D@!?G?`??D@R      ??!       Z	?G?`??D@?G?`??D@!?G?`??D@b      ??!       JCPU_ONLYYL?D?8@b q???ku|W@