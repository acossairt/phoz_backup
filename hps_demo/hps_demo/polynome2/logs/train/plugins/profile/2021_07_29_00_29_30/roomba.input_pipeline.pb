	??=Ab?G@??=Ab?G@!??=Ab?G@	?B??????B?????!?B?????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??=Ab?G@??lɪP$@AE?[???B@Y?-???=??*	-???Sh@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???;?_??!?L;??{>@)?ۼqR???10P???7@:Preprocessing2F
Iterator::Model?b??	???!	ߙ???D@)f3??J??1?
??M^6@:Preprocessing2U
Iterator::Model::ParallelMapV2ޑ??????!???+??2@)ޑ??????1???+??2@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat;?vٯ;??!,e??N5@)
???????1U?h??1@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicex'???!???77@)x'???1???77@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip_Cp\?M??!? fZhM@)*???;??1?CM@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?K?b?{?!T?e?
@)?K?b?{?1T?e?
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?Qԙ{H??!??-@W@@)?n???q?1?V?Η@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 21.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?B?????I??H?	?X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??lɪP$@??lɪP$@!??lɪP$@      ??!       "      ??!       *      ??!       2	E?[???B@E?[???B@!E?[???B@:      ??!       B      ??!       J	?-???=???-???=??!?-???=??R      ??!       Z	?-???=???-???=??!?-???=??b      ??!       JCPU_ONLYY?B?????b q??H?	?X@