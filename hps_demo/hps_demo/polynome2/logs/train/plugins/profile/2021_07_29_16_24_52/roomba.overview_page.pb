?	?(^e??d@?(^e??d@!?(^e??d@	;??E??;??E??!;??E??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?(^e??d@|'f?>;@A5?\??a@Y?&S???*	?(\???@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?????4@!迟???X@)F?xx
4@1絮??X@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?X????!N
?Q???)?X????1N
?Q???:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???6????!???I\??).Y?&???1????l??:Preprocessing2F
Iterator::Model?p?q?t??!?*???9??)??p<???1*f?L????:Preprocessing2U
Iterator::Model::ParallelMapV2Y"?????!Yﲄ!???)Y"?????1Yﲄ!???:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?je?/14@!?A??X@)?W?\??1?[??G??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorM?~2Ƈ??!??/?|??)M?~2Ƈ??1??/?|??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapu?V4@!n?[???X@)&:?,B?u?1?N?k???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 16.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9;??E??I??R?n?X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	|'f?>;@|'f?>;@!|'f?>;@      ??!       "      ??!       *      ??!       2	5?\??a@5?\??a@!5?\??a@:      ??!       B      ??!       J	?&S????&S???!?&S???R      ??!       Z	?&S????&S???!?&S???b      ??!       JCPU_ONLYY;??E??b q??R?n?X@Y      Y@q???????"?
both?Your program is POTENTIALLY input-bound because 16.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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