?	46<=_o@46<=_o@!46<=_o@	_??e<?@_??e<?@!_??e<?@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$46<=_o@N??ĭ??A8M?p?m@Yb??c?,@*	?A`???@2F
Iterator::Model?p?{??,@!?]?ǿN@)?4???+@1=\)?_M@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?u???!@!?P?C@)?]ؚ??!@1??6??B@:Preprocessing2U
Iterator::Model::ParallelMapV2f???8???!??݁V
@)f???8???1??݁V
@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatbhur????!!??M??)Um7?7M??1!ACn???:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicea???U??!?i5)??)a???U??1?i5)??:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??d?z?!@!g?? 8@C@)@?P???1 ??%?A??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??͎T߉?!?:?ϫ?)??͎T߉?1?:?ϫ?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?X??!@!??:P?C@)????w?1wք?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 5.7% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no9`??e<?@I??9ܑW@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	N??ĭ??N??ĭ??!N??ĭ??      ??!       "      ??!       *      ??!       2	8M?p?m@8M?p?m@!8M?p?m@:      ??!       B      ??!       J	b??c?,@b??c?,@!b??c?,@R      ??!       Z	b??c?,@b??c?,@!b??c?,@b      ??!       JCPU_ONLYY`??e<?@b q??9ܑW@Y      Y@qRr?C??|?"?
both?Your program is MODERATELY input-bound because 5.7% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
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