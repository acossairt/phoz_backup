?	?N???J@?N???J@!?N???J@	z?ӡ????z?ӡ????!z?ӡ????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?N???J@??x?@e,@AQ?|a?C@Y+Kt?Y???*	v??/?d@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatb?k_@??!a?G_?i;@)g??e???1Ϟ8??^6@:Preprocessing2F
Iterator::Model??	?8??!=N?,C@)?#?????1??,??3@:Preprocessing2U
Iterator::Model::ParallelMapV2??~?Ϛ??!? 	?u?2@)??~?Ϛ??1? 	?u?2@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::ConcatenateZ?!?[=??!ԡ?}/f;@)eM??1?????1@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice??<??-??!???#@)??<??-??1???#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?&??0??!ñl???N@)Y???j??1-??F??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??S????!FN=<?+@)??S????1FN=<?+@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapJ)???ƨ?!Z????5=@)?~j?t?h?1]?D????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 26.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9z?ӡ????I"???X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??x?@e,@??x?@e,@!??x?@e,@      ??!       "      ??!       *      ??!       2	Q?|a?C@Q?|a?C@!Q?|a?C@:      ??!       B      ??!       J	+Kt?Y???+Kt?Y???!+Kt?Y???R      ??!       Z	+Kt?Y???+Kt?Y???!+Kt?Y???b      ??!       JCPU_ONLYYz?ӡ????b q"???X@Y      Y@q?s???7@"?	
both?Your program is POTENTIALLY input-bound because 26.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?23.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 