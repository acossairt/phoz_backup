	??????)@??????)@!??????)@	o?H?l?@o?H?l?@!o?H?l?@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??????)@V???n?@A	??g??@Y?lV}.??*	?"???2?@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?R	???!??k9T@)?6ɏ???1c?0m??S@:Preprocessing2U
Iterator::Model::ParallelMapV2????|???!?.Eyc?@)????|???1?.Eyc?@:Preprocessing2F
Iterator::Modelhur?????!S?(???(@)"P??H???1}N@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatLR?b???!
x?´e@)??????1\?Ѫm@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?Y?X"??!???X?j??)?Y?X"??1???X?j??:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipq?Qe7??!v???$?U@)??|ԛ??1????<d??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?B?_?+??!?O???v??)?B?_?+??1?O???v??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap.?ED1???!Z?Р8TT@)??H??u?1?B?<????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 47.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9o?H?l?@Iu6Y?W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	V???n?@V???n?@!V???n?@      ??!       "      ??!       *      ??!       2		??g??@	??g??@!	??g??@:      ??!       B      ??!       J	?lV}.???lV}.??!?lV}.??R      ??!       Z	?lV}.???lV}.??!?lV}.??b      ??!       JCPU_ONLYYo?H?l?@b qu6Y?W@