	???l?nT@???l?nT@!???l?nT@	`?~c???`?~c???!`?~c???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???l?nT@??m??4@A?蜟?tN@Y???ʦ??*	ףp=*h@2U
Iterator::Model::ParallelMapV2c?ZB>???!??O??,:@)c?ZB>???1??O??,:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat2???????!Ƒ@ݏ?9@)?س?2??1n???j5@:Preprocessing2F
Iterator::Modelx}??O9??!tpU?sF@)?ra???1D [?L?2@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate?????I??!"?h???6@)/??$???1?=6 |;-@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice??OI??!ou7	??@)??OI??1ou7	??@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?*øD??!?????K@)İØ????1??R**@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?$\?#???!\?M???@)?$\?#???1\?M???@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?e?I)??!???	i8@)????a?m?1l?n?rF??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 25.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9`?~c???I?@????X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??m??4@??m??4@!??m??4@      ??!       "      ??!       *      ??!       2	?蜟?tN@?蜟?tN@!?蜟?tN@:      ??!       B      ??!       J	???ʦ?????ʦ??!???ʦ??R      ??!       Z	???ʦ?????ʦ??!???ʦ??b      ??!       JCPU_ONLYY`?~c???b q?@????X@