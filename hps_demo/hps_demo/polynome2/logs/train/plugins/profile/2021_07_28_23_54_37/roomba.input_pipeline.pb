	?????Ez@?????Ez@!?????Ez@	?$?????$????!?$????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?????Ez@?.????A$???t@z@Y!???3ں?*	X9??~?@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate?????Q??!A????S@)?T????1-??q?cS@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?0?????!Z??G["@)?ަ?????1??6A.?@:Preprocessing2U
Iterator::Model::ParallelMapV2X??0_^??!?????@)X??0_^??1?????@:Preprocessing2F
Iterator::Model?l?M??!?S???!@)??Ry=??1?@?f?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?;?(A???!#?5???V@)??E|'f??1?|????:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice?k
dv??!x?"?????)?k
dv??1x?"?????:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?$???7??!ǧt9!???)?$???7??1ǧt9!???:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap8h??s??!;lޡ?T@)L7?A`?p?1?	SV??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?$????I??/g?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?.?????.????!?.????      ??!       "      ??!       *      ??!       2	$???t@z@$???t@z@!$???t@z@:      ??!       B      ??!       J	!???3ں?!???3ں?!!???3ں?R      ??!       Z	!???3ں?!???3ں?!!???3ں?b      ??!       JCPU_ONLYY?$????b q??/g?X@