	??Q?	o@??Q?	o@!??Q?	o@	??⵼(????⵼(??!??⵼(??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??Q?	o@u?i????A?|a2?n@Y???D
@*	u???@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateJC?B@!lF3[|iK@)[A?+?@1??}ވ1K@:Preprocessing2F
Iterator::Model??????	@!$???2F@)?!9??@1??8HB@:Preprocessing2U
Iterator::Model::ParallelMapV2?N?P\??!???^? @)?N?P\??1???^? @:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?,{ؤ?!???J????)?G??Q??1?6Zݦ??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice????V??!uE?Z????)????V??1uE?Z????:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??)X?<@!?Ap
??K@)??i????1FC?"???:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorY?;ۣ7|?!??z??(??)Y?;ۣ7|?1??z??(??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap9~?4b@!?T#??pK@)??Д?~p?1??9?k>??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??⵼(??Idt(]?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	u?i????u?i????!u?i????      ??!       "      ??!       *      ??!       2	?|a2?n@?|a2?n@!?|a2?n@:      ??!       B      ??!       J	???D
@???D
@!???D
@R      ??!       Z	???D
@???D
@!???D
@b      ??!       JCPU_ONLYY??⵼(??b qdt(]?X@