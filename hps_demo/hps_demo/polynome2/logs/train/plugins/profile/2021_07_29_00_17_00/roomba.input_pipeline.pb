	??TNg@??TNg@!??TNg@	?@0??P???@0??P??!?@0??P??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??TNg@	?3????AM????f@Y??:8؛??*	Zd;?Ok@2U
Iterator::Model::ParallelMapV25?磌??!.???6@)5?磌??1.???6@:Preprocessing2F
Iterator::Model??A_z???!Pԍ?E@)?(??Pj??1r?k?Q5@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?y?):???!r?kE?8@)o??\????1?=?&<R4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate$+?ƈ??!!w?YC?9@)S??%?Ѣ?1OϞ???0@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice????qn??!?O;<!@)????qn??1?O;<!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???6T???!?+r??gL@)?z?2Q???1?,??7y@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???3K??!??-%@)???3K??1??-%@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[??Ye??!???l`^;@)??Ң>?m?1?x?2????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?@0??P??I?s??k?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
		?3????	?3????!	?3????      ??!       "      ??!       *      ??!       2	M????f@M????f@!M????f@:      ??!       B      ??!       J	??:8؛????:8؛??!??:8؛??R      ??!       Z	??:8؛????:8؛??!??:8؛??b      ??!       JCPU_ONLYY?@0??P??b q?s??k?X@