?	?w???@?w???@!?w???@	??[F֣???[F֣?!??[F֣?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?w???@??????A??փ?@YB!?J??*	Zd;?#s@2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice	?3????!?\z???1@)	?3????1?\z???1@:Preprocessing2U
Iterator::Model::ParallelMapV2\?3??O??!??J?j1@)\?3??O??1??J?j1@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat+??-??!?ZY?a?9@)??M?q??1d???ߥ0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate?M~?N???!??v???@@)??u?ݑ??14vs??N0@:Preprocessing2F
Iterator::Model?:␹?!?oQ[$N@@)?R???ҧ?1?9???b.@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor.??????!IvCA/"@).??????1IvCA/"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?"??~j??!=HW???P@)???;3??1W??ك?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?r?9>Z??!??;B@)?J???>|?1*1??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??[F֣?I??47??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????????????!??????      ??!       "      ??!       *      ??!       2	??փ?@??փ?@!??փ?@:      ??!       B      ??!       J	B!?J??B!?J??!B!?J??R      ??!       Z	B!?J??B!?J??!B!?J??b      ??!       JCPU_ONLYY??[F֣?b q??47??X@Y      Y@q?s?]D@"?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
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