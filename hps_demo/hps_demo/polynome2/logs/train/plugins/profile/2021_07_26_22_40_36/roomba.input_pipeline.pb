	??HhJ@??HhJ@!??HhJ@	4???/@4???/@!4???/@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??HhJ@τ&?%??A9?j?3%F@Y$Dib @*	;?O?7?@2F
Iterator::Model獓¼G @!??m?b1Q@)?**??@1??L??C@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenates???@!??Tv?>@)鷯?@1-[]?d>@:Preprocessing2U
Iterator::Model::ParallelMapV2????@!??o?5=@)????@1??o?5=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatD?Ac&??!֌?U???)??a?Q+??1=x6 ?L??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??????!?.?}P\??)??????1?.?}P\??:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???D?@!?H?t:?@)E??2???1BG[ ????:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor
?F???!aR??3
??)
?F???1aR??3
??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?bb?q@!0???e?>@)???M??p?1?
 ?ߡ?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 15.5% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no93???/@Iz??&AU@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	τ&?%??τ&?%??!τ&?%??      ??!       "      ??!       *      ??!       2	9?j?3%F@9?j?3%F@!9?j?3%F@:      ??!       B      ??!       J	$Dib @$Dib @!$Dib @R      ??!       Z	$Dib @$Dib @!$Dib @b      ??!       JCPU_ONLYY3???/@b qz??&AU@