?	D?1uW?g@D?1uW?g@!D?1uW?g@	?eG?@?eG?@!?eG?@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$D?1uW?g@f??ᔹ??AS"?^F_f@Yro~?Dc'@*	:??v??@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?YO?&%@!?X?K[?G@)??̓k%@1???pe?G@:Preprocessing2U
Iterator::Model::ParallelMapV2????#@!??:@ǝF@)????#@1??:@ǝF@:Preprocessing2F
Iterator::Modelӈ?}?&@!XF??I@)K9_?????1???.?	@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?H?+???!?=?A????)??5&Ĥ?1]r@?????:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSliceʨ2??A??!???????)ʨ2??A??1???????:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???)?I%@!????!H@)?Z'.?+??1???MT??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?΢w*?~?!?-	???)?΢w*?~?1?-	???:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????(%@!?b!??G@)0?a?[>r?1??O?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 6.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no9?eG?@I???xW@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	f??ᔹ??f??ᔹ??!f??ᔹ??      ??!       "      ??!       *      ??!       2	S"?^F_f@S"?^F_f@!S"?^F_f@:      ??!       B      ??!       J	ro~?Dc'@ro~?Dc'@!ro~?Dc'@R      ??!       Z	ro~?Dc'@ro~?Dc'@!ro~?Dc'@b      ??!       JCPU_ONLYY?eG?@b q???xW@Y      Y@q!$?E?Q??"?
both?Your program is MODERATELY input-bound because 6.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"CPU: B 