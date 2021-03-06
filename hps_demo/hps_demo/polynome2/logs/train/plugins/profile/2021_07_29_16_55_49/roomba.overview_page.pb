?	??.??b@??.??b@!??.??b@	?????9@?????9@!?????9@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??.??b@??z?<???A??7??=a@YG?j???&@*	??K7)??@2U
Iterator::Model::ParallelMapV2G?ŧ X@!??g?] D@)G?ŧ X@1??g?] D@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?
)???@!?'XH7@)J	??z?@1??i33?6@:Preprocessing2F
Iterator::Model?>rk?}&@!?M??f?N@)??=^HG@1Bl]t5@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?z?<d?@!Z?iO?%C@)&m???M??1H??us
"@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???0??!u?v???@)??/???1??z練@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice3??(]??!r[+?$??)3??(]??1r[+?$??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor!yv???!{f?????)!yv???1{f?????:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?Y????@!?k?	7@).?;1??p?1tU6?C??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 7.6% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no9?????9@I???bbW@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??z?<?????z?<???!??z?<???      ??!       "      ??!       *      ??!       2	??7??=a@??7??=a@!??7??=a@:      ??!       B      ??!       J	G?j???&@G?j???&@!G?j???&@R      ??!       Z	G?j???&@G?j???&@!G?j???&@b      ??!       JCPU_ONLYY?????9@b q???bbW@Y      Y@q???M@??"?
both?Your program is MODERATELY input-bound because 7.6% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
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