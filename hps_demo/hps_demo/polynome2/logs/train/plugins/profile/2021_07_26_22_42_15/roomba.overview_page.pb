?	կt>?^f@կt>?^f@!կt>?^f@	?\??a?@?\??a?@!?\??a?@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$կt>?^f@,?/o?9@A?H???*b@YP?mp?@*3^?ILj?@)      p=2F
Iterator::Model???|?@!Mq?3W@)˃?9?@1Tj?j?bU@:Preprocessing2U
Iterator::Model::ParallelMapV2/?h?R??!|oP"x@)/?h?R??1|oP"x@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??c${???!?n?k@)I???p???1??"^@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice,??f*ģ?!P???g??),??f*ģ?1P???g??:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat=_?\6:??!Y??j=G??)D? ????1O~??Q7??:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?n?KS???!9??.??@)?(??0??1???{T???:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?\?	?}?!T?>]??)?\?	?}?1T?>]??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapQ???????!?i?ݖ?@)??{???s?1O???{^??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 14.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?\??a?@I6?5???W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	,?/o?9@,?/o?9@!,?/o?9@      ??!       "      ??!       *      ??!       2	?H???*b@?H???*b@!?H???*b@:      ??!       B      ??!       J	P?mp?@P?mp?@!P?mp?@R      ??!       Z	P?mp?@P?mp?@!P?mp?@b      ??!       JCPU_ONLYY?\??a?@b q6?5???W@Y      Y@q??if̥?"?
both?Your program is POTENTIALLY input-bound because 14.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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