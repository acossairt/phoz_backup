	?s??Q??@?s??Q??@!?s??Q??@	??c?@??c?@!??c?@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?s??Q??@Ϡ?????Aw?x?6??@Y?_?L?P@*	V?J A2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate?Z`???P@!??Ă??H@)ap???P@1hY.kQ?H@:Preprocessing2U
Iterator::Model::ParallelMapV2{m?ԏP@!)l|??H@){m?ԏP@1)l|??H@:Preprocessing2F
Iterator::Model??ypw?P@!?>??H@)?#?????1?Z?h?4??:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?ĭ????!a?$??)??)?'???I??1?Q?y???:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice?.?.ǘ?!O?㴼???)?.?.ǘ?1O?㴼???:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??T?P@!7??~?I@)??p?Qe??1yG??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??ԕ????!T?ܿ???)??ԕ????1T?ܿ???:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapa?٨P@!?/??H@)?S???
t?1?z???n?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 7.5% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no9??c?@I_D??4W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Ϡ?????Ϡ?????!Ϡ?????      ??!       "      ??!       *      ??!       2	w?x?6??@w?x?6??@!w?x?6??@:      ??!       B      ??!       J	?_?L?P@?_?L?P@!?_?L?P@R      ??!       Z	?_?L?P@?_?L?P@!?_?L?P@b      ??!       JCPU_ONLYY??c?@b q_D??4W@