?	 ??4v?s@ ??4v?s@! ??4v?s@	?eޱ?@?eޱ?@!?eޱ?@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$ ??4v?s@?J"? ???AS??Ʈ?r@Y4h????2@*	?l??)??@2F
Iterator::Model???ӝ?@!)??k+P@)2t??@1??*???O@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate4??X?_??!?ˑ???@@)?D??b??1g??@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?,??V??!?n????)??p<???1lt'??F??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?Ov3???!a?????)?Ov3???1a?????:Preprocessing2U
Iterator::Model::ParallelMapV2\Y???"??!+?;?????)\Y???"??1+?;?????:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??V`?*??!?UT(??A@);TS?u8??1-??3???:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?b.???!?=;*?7??)?b.???1?=;*?7??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??9z|??!r???@@)??D???|?1j??}WV??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 5.9% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no9?eޱ?@I???ăW@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?J"? ????J"? ???!?J"? ???      ??!       "      ??!       *      ??!       2	S??Ʈ?r@S??Ʈ?r@!S??Ʈ?r@:      ??!       B      ??!       J	4h????2@4h????2@!4h????2@R      ??!       Z	4h????2@4h????2@!4h????2@b      ??!       JCPU_ONLYY?eޱ?@b q???ăW@Y      Y@qXAi????"?
both?Your program is MODERATELY input-bound because 5.9% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
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