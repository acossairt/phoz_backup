?	"ĕ??\@"ĕ??\@!"ĕ??\@	?6?85>???6?85>??!?6?85>??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$"ĕ??\@??.%@Az?蹅Z@Yr?t?????*	????K?{@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::ConcatenateN)???]??!?\?]bM@)R?r????1?????@J@:Preprocessing2F
Iterator::Model??uS?k??!??*?A?6@)1?Tm7???1Q??zm%,@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatL?^I???!+?Rp?
.@)y?ՏM???1~?L,M?(@:Preprocessing2U
Iterator::Model::ParallelMapV2???9]??!|ݍA? @)???9]??1|ݍA? @:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceMK?????!֧?-S@)MK?????1֧?-S@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipa7l[????!Ru??]S@)Na??????1?~9T@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorx?g?ɇ?!??!@)x?g?ɇ?1??!@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?ѯ????!??Eq$xM@)e?z?Fwp?1O=^??0??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 9.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?6?85>??Ie?c?`?X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??.%@??.%@!??.%@      ??!       "      ??!       *      ??!       2	z?蹅Z@z?蹅Z@!z?蹅Z@:      ??!       B      ??!       J	r?t?????r?t?????!r?t?????R      ??!       Z	r?t?????r?t?????!r?t?????b      ??!       JCPU_ONLYY?6?85>??b qe?c?`?X@Y      Y@q0?dA@"?	
both?Your program is POTENTIALLY input-bound because 9.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?34.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 