?	?ܘ?p??@?ܘ?p??@!?ܘ?p??@	=ֱ?g@=ֱ?g@!=ֱ?g@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?ܘ?p??@V?p;4,??A?S[
~?@Y t?//p-@*	VE??@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???hȨ"@!??Wj]C@)JF??"@1#NP?RC@:Preprocessing2U
Iterator::Model::ParallelMapV2H?'ۈ"@!J?7H<C@)H?'ۈ"@1J?7H<C@:Preprocessing2F
Iterator::Model?\?B-@!$<8^N@)??&k?s@1?h?C6@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??x?Z???!?&??|???) ??Ud??1ߟ???)??:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipE??f??"@!??????C@)??DJ?y??1I??[???:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice>?n?KS??!?Lש??)>?n?KS??1?Lש??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?{+Ԑ?!
??v??)?{+Ԑ?1
??v??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?m??f?"@!??? "`C@)0?[w?t?1?%??H???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 2.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9>ֱ?g@I?Oq??dX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	V?p;4,??V?p;4,??!V?p;4,??      ??!       "      ??!       *      ??!       2	?S[
~?@?S[
~?@!?S[
~?@:      ??!       B      ??!       J	 t?//p-@ t?//p-@! t?//p-@R      ??!       Z	 t?//p-@ t?//p-@! t?//p-@b      ??!       JCPU_ONLYY>ֱ?g@b q?Oq??dX@Y      Y@q_???|?"?
device?Your program is NOT input-bound because only 2.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
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