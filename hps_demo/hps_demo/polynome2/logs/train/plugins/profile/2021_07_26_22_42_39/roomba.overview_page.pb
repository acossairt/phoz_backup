?	C9Ѯ"X?@C9Ѯ"X?@!C9Ѯ"X?@	??B?????B???!??B???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$C9Ѯ"X?@??S?????A??gy.??@Y??:?6@*	+?9??@2F
Iterator::Model???C?,@!????\[U@){?2Q?t(@1qg6?PR@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatez6?>W@!?ɎPTC,@)??Ry?@1?͢?z?+@:Preprocessing2U
Iterator::Model::ParallelMapV2??C??@!???U`(*@)??C??@1???U`(*@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicej?֍w??!?|?8[??)j?֍w??1?|?8[??:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatq;4,F]??!eC?m$=??)e?????1@I'????:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor~p>u???!???gO???)~p>u???1???gO???:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??U??@!?;y?%-@)c?#?w~??1??h????:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap8??L?#@!~X<XoO,@)?|	^p?1?i[6??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??B???I??ҡ?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??S???????S?????!??S?????      ??!       "      ??!       *      ??!       2	??gy.??@??gy.??@!??gy.??@:      ??!       B      ??!       J	??:?6@??:?6@!??:?6@R      ??!       Z	??:?6@??:?6@!??:?6@b      ??!       JCPU_ONLYY??B???b q??ҡ?X@Y      Y@q??y?1p?"?
device?Your program is NOT input-bound because only 1.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
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