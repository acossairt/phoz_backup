	?ʃ?i6@?ʃ?i6@!?ʃ?i6@	 ?L
??? ?L
???! ?L
???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?ʃ?i6@_
?]?@A>v()?1@Y?I?2???*x?G??g@)       =2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??|#?g??!?2{X%>@)зKu??1Y???{?7@:Preprocessing2U
Iterator::Model::ParallelMapV2w?Nyt#??!Dy?
G?4@)w?Nyt#??1Dy?
G?4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??F??!?z?Ku?7@)Kt?Y?b??1u?"_??3@:Preprocessing2F
Iterator::ModelY??9?}??!?Mș??C@);??]آ?1!"?(nI3@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???a???!i??2@)???a???1i??2@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip>?٬?\??!N?7f%N@)?3????1?r??v@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorjkD0.}?!\ӿc?@)jkD0.}?1\ӿc?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?eS????! MWO??@)~Q??B?h?1d????"??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 19.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9 ?L
???I?#f???X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	_
?]?@_
?]?@!_
?]?@      ??!       "      ??!       *      ??!       2	>v()?1@>v()?1@!>v()?1@:      ??!       B      ??!       J	?I?2????I?2???!?I?2???R      ??!       Z	?I?2????I?2???!?I?2???b      ??!       JCPU_ONLYY ?L
???b q?#f???X@